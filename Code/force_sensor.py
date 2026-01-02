# force_sensor.py
#
# Single-shot: one reference image + one deformed image -> FTP height (mm) -> volume (cm^3) -> force (N)

from __future__ import annotations

import os
import json
import csv
from typing import Dict, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

import shape_ftp


# ===========================
# PATHS 
# ===========================
REFERENCE_PATH = "./Final_demos_images/FINAL_reference.jpg"
DEFORMED_PATH= "./Final_demos_images/FINAL_E_deformed.jpg" 
OUTPUT_DIR     = "./Force/force_sensor_out"

# Force calibration JSON from height_to_force.py
FORCE_CALIBRATION_JSON = "./Force/Height_to_force/calibration_out/calibration_model.json"


# ===========================
# CONFIG
# ===========================
GRATING_PITCH_MM = 2.0      # grating pitch (mm)
DEPTH_EPS_MM     = 0.01     # integrate depth > eps

OVERRIDE_MM_PER_PX = None  

SHAPE_CALIBRATION_JSON = getattr(shape_ftp, "CALIBRATION_JSON", None)
SAVE_SUMMARY_FIGURES = True
EXPORT_HEIGHTMAPS = False
DEBUG = False

# Interactive window toggle
SHOW_3D_HEIGHTMAP_POPUP = True


# ===========================
# UTIL
# ===========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def safe_float(x, fallback=np.nan):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(fallback)


# ===========================
# 3D PLOT (same as shape_ftp.py)
# ===========================
def plot_height_map_interactive(height_map, circ_mask=None, title="Height map (interactive 3D)"):
    h, w = height_map.shape
    Y, X = np.mgrid[0:h, 0:w]
    Z = height_map.astype(float).copy()
    if circ_mask is not None:
        Z[~circ_mask] = np.nan

    step = max(1, int(min(h, w) / 350))
    X_ds = X[::step, ::step]
    Y_ds = Y[::step, ::step]
    Z_ds = Z[::step, ::step]

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    Zm = np.ma.masked_invalid(Z_ds)
    surf = ax.plot_surface(X_ds, Y_ds, Zm, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_zlabel("height (mm)")
    fig.colorbar(surf, shrink=0.6, label="mm")
    return fig


# ===========================
# VOLUME INTEGRATION (same as height_to_force.py)
# ===========================
def depth_map_to_volume_cm3(
    height_map_mm: np.ndarray,
    roi_mask: np.ndarray,
    mm_per_px: float,
    depth_eps_mm: float = 0.01,
) -> Tuple[float, float, float]:
    """
    Integrate V = sum(depth_mm * pixel_area_mm2) over depth>eps within ROI.
    Returns: (volume_cm3, contact_area_mm2, max_depth_mm)
    """
    Z = np.asarray(height_map_mm, dtype=np.float32).copy()
    roi = np.asarray(roi_mask, dtype=bool)

    pos = np.clip(Z, 0.0, np.inf)
    neg = np.clip(-Z, 0.0, np.inf)
    depth = neg if float(np.nansum(neg)) > float(np.nansum(pos)) else pos

    depth[~roi] = 0.0
    depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)

    contact = depth > float(depth_eps_mm)
    if not np.any(contact):
        return 0.0, 0.0, 0.0

    pixel_area_mm2 = float(mm_per_px) ** 2
    volume_mm3 = float(np.sum(depth[contact]) * pixel_area_mm2)
    area_mm2 = float(np.count_nonzero(contact) * pixel_area_mm2)
    max_depth_mm = float(np.max(depth[contact]))

    volume_cm3 = volume_mm3 / 1000.0
    return float(volume_cm3), float(area_mm2), float(max_depth_mm)


# ===========================
# FORCE MODEL (from calibration_model.json)
# ===========================
def _m_linear0(v, a): return a * v
def _m_linear(v, a, b): return a * v + b
def _m_poly2(v, c2, c1, c0): return c2 * v * v + c1 * v + c0
def _m_sat_exp(v, a, b): return a * (1.0 - np.exp(-b * np.maximum(v, 0.0)))
def _m_growth(v, a, b): return a * (np.exp(b * np.maximum(v, 0.0)) - 1.0)

def _m_hinge_sat(v, a, b, c):
    vv = np.asarray(v, float)
    return a * (
        (1.0 - np.exp(-b * np.maximum(vv - c, 0.0)))
        - (1.0 - np.exp(-b * np.maximum(0.0 - c, 0.0)))
    )

def load_force_calibration(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "best_model" not in data:
        raise ValueError("Invalid force calibration JSON: missing 'best_model'")
    return data

def predict_force_from_volume(best_model: Dict[str, Any], volume_cm3: float) -> float:
    t = best_model["type"]
    p = best_model["params"]
    v = float(volume_cm3)

    if t == "linear0":
        return float(_m_linear0(v, float(p["a"])))
    if t == "linear":
        return float(_m_linear(v, float(p["a"]), float(p["b"])))
    if t == "poly2":
        return float(_m_poly2(v, float(p["c2"]), float(p["c1"]), float(p["c0"])))
    if t == "sat_exp":
        return float(_m_sat_exp(v, float(p["a"]), float(p["b"])))
    if t == "growth":
        return float(_m_growth(v, float(p["a"]), float(p["b"])))
    if t == "hinge_saturating":
        return float(_m_hinge_sat(v, float(p["a"]), float(p["b"]), float(p["c"])))

    raise ValueError(f"Unknown model type in force calibration JSON: {t}")


# ===========================
# SCALE (mm/px)
# ===========================
def estimate_mm_per_px(estimated_grating_period_px: Optional[float]) -> float:
    if OVERRIDE_MM_PER_PX is not None:
        return float(OVERRIDE_MM_PER_PX)

    if estimated_grating_period_px is None:
        raise RuntimeError("shape_ftp did not return estimated_grating_period_px and OVERRIDE_MM_PER_PX is not set.")

    est = float(estimated_grating_period_px)
    if (not np.isfinite(est)) or est <= 1e-12:
        raise RuntimeError(
            f"Invalid estimated_grating_period_px={estimated_grating_period_px}. "
            "Fix shape_ftp return or set OVERRIDE_MM_PER_PX."
        )

    return float(GRATING_PITCH_MM) / est


# ===========================
# MAIN
# ===========================
def main():
    ensure_dir(OUTPUT_DIR)

    if not os.path.isfile(REFERENCE_PATH):
        raise FileNotFoundError(f"REFERENCE_PATH not found: {REFERENCE_PATH}")
    if not os.path.isfile(DEFORMED_PATH):
        raise FileNotFoundError(f"DEFORMED_PATH not found: {DEFORMED_PATH}")
    if not os.path.isfile(FORCE_CALIBRATION_JSON):
        raise FileNotFoundError(f"FORCE_CALIBRATION_JSON not found: {FORCE_CALIBRATION_JSON}")

    force_calib = load_force_calibration(FORCE_CALIBRATION_JSON)
    best_model = force_calib["best_model"]

    # Where shape_ftp can write its run artifacts
    ftp_out_dir = os.path.join(OUTPUT_DIR, "ftp_run")
    ensure_dir(ftp_out_dir)

    # Run FTP -> height map (mm)
    res = shape_ftp.main(
        reference_path=REFERENCE_PATH,
        deformed_path=DEFORMED_PATH,
        output_dir=ftp_out_dir,
        calibration_json=SHAPE_CALIBRATION_JSON,
        batch_mode=True, 
        save_summary_figures=SAVE_SUMMARY_FIGURES,
        export_heightmaps=EXPORT_HEIGHTMAPS,
        debug=DEBUG,
        return_results=True,
    )

    height_mm = res["height_map_mm_crop"]
    roi = res["roi_eroded_crop"]
    est_period_px = res.get("estimated_grating_period_px", None)

    if SHOW_3D_HEIGHTMAP_POPUP:
        plot_height_map_interactive(height_mm, circ_mask=roi, title="Height map (3D) - force_sensor")
        plt.show()

    mm_per_px = estimate_mm_per_px(est_period_px)

    volume_cm3, area_mm2, max_depth_mm = depth_map_to_volume_cm3(
        height_map_mm=height_mm,
        roi_mask=roi,
        mm_per_px=mm_per_px,
        depth_eps_mm=DEPTH_EPS_MM,
    )

    force_n = predict_force_from_volume(best_model, volume_cm3)

    out = {
        "reference_path": REFERENCE_PATH,
        "deformed_path": DEFORMED_PATH,
        "output_dir": OUTPUT_DIR,
        "ftp_output_dir": ftp_out_dir,
        "grating_pitch_mm": float(GRATING_PITCH_MM),
        "depth_eps_mm": float(DEPTH_EPS_MM),
        "estimated_grating_period_px": None if est_period_px is None else safe_float(est_period_px, np.nan),
        "mm_per_px": float(mm_per_px),
        "volume_cm3": float(volume_cm3),
        "contact_area_mm2": float(area_mm2),
        "max_depth_mm": float(max_depth_mm),
        "force_N": float(force_n),
        "force_model": {
            "type": best_model.get("type", ""),
            "params": best_model.get("params", {}),
            "equation": best_model.get("equation", ""),
            "rmse": best_model.get("rmse", None),
            "r2": best_model.get("r2", None),
        },
    }

    json_path = os.path.join(OUTPUT_DIR, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    csv_path = os.path.join(OUTPUT_DIR, "result.csv")
    fieldnames = [
        "reference_path",
        "deformed_path",
        "volume_cm3",
        "force_N",
        "contact_area_mm2",
        "max_depth_mm",
        "mm_per_px",
        "estimated_grating_period_px",
        "ftp_output_dir",
        "force_model_type",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow({
            "reference_path": REFERENCE_PATH,
            "deformed_path": DEFORMED_PATH,
            "volume_cm3": float(volume_cm3),
            "force_N": float(force_n),
            "contact_area_mm2": float(area_mm2),
            "max_depth_mm": float(max_depth_mm),
            "mm_per_px": float(mm_per_px),
            "estimated_grating_period_px": None if est_period_px is None else safe_float(est_period_px, np.nan),
            "ftp_output_dir": ftp_out_dir,
            "force_model_type": best_model.get("type", ""),
        })

    print(f"volume_cm3    = {volume_cm3:.6g}")
    print(f"force_N       = {force_n:.6g}")
    print(f"max_depth_mm  = {max_depth_mm:.6g}")
    print(f"saved         = {json_path}")
    print(f"saved         = {csv_path}")


if __name__ == "__main__":
    main()
