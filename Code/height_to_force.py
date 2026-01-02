# height_to_force.py
#
# Batch-calibrate: indentation VOLUME (from FTP height map) -> FORCE (Newtons)
#
# Dataset layout:
#   - 15 force levels: [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15, 20, 25, 30, 35, 40, 45]
#   - 5 images per force level, ordered sequentially:
#       sphere-1.jpg ... sphere-75.jpg
#       (images 1..5 -> 0.5N, 6..10 -> 1.0N, ..., 71..75 -> 45N)
#
# What it does:
#   1) For each image, calls shape_ftp.py to compute a DEPTH map in mm (same FTP pipeline).
#   2) Estimates XY scale (mm/px) from the grating pitch and the FFT carrier (returned by shape_ftp).
#   3) Integrates volume:  V = sum(depth_mm * pixel_area_mm2) over depth>DEPTH_EPS_MM in ROI
#      Outputs in cm^3.
#   4) Saves:
#        - per_image_results.csv
#        - calibration_model.json  (best model: F = f(V))
#        - force_vs_volume_boxplot_and_fit.png  (boxplot per force + scatter+fit)


import os
import json
import csv
import numpy as np
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import shape_ftp


# ===========================
# PATHS (EDIT THESE)
# ===========================
REFERENCE_PATH = "./Force/FINAL_reference.jpg"
DEFORMED_DIR   = "./Force/Height_to_force/Loading"
OUTPUT_DIR     = "./Force/Height_to_force/calibration_out"

# ===========================
# DATASET DEFINITION
# ===========================
FORCE_LEVELS_N = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
IMAGES_PER_LEVEL = 5
IMAGE_PATTERN = "sphere-{}.jpg"
IMAGE_START_INDEX = 1  # sphere-1.jpg


# ===========================
# VOLUME INTEGRATION CONFIG
# ===========================
# Your grating stripe width is 1 mm, period is 2.0 mm (black+white).
GRATING_PITCH_MM = 2.0

# Only integrate pixels deeper than this (mm)
DEPTH_EPS_MM = 0.01


OVERRIDE_MM_PER_PX = None


# ===========================
# MODEL FIT CONFIG (force = f(volume))
# ===========================
ANCHOR_ORIGIN = True
ORIGIN_WEIGHT = 20  # repeats (0,0) this many times (helps make F(0)=0)

MODEL_CANDIDATES = [
    "linear0",          # F = a * V
    "linear",           # F = a * V + b
    "poly2",            # F = c2*V^2 + c1*V + c0
    "sat_exp",          # F = a * (1 - exp(-b*V))
    "growth",           # F = a * (exp(b*V) - 1)
    "hinge_saturating", 
]


# ===========================
# UTIL
# ===========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-18:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def safe_float(x, fallback=np.nan):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(fallback)


# ===========================
# MODELS: force = f(volume)
# ===========================
def _m_linear0(v, a):
    return a * v

def _m_linear(v, a, b):
    return a * v + b

def _m_poly2(v, c2, c1, c0):
    return c2 * v * v + c1 * v + c0

def _m_sat_exp(v, a, b):
    # a*(1-exp(-b*v)) ensures F(0)=0
    return a * (1.0 - np.exp(-b * np.maximum(v, 0.0)))

def _m_growth(v, a, b):
    # a*(exp(b*v)-1) ensures F(0)=0
    return a * (np.exp(b * np.maximum(v, 0.0)) - 1.0)

def _m_hinge_sat(v, a, b, c):
    vv = np.asarray(v, float)
    return a * (
        (1.0 - np.exp(-b * np.maximum(vv - c, 0.0)))
        - (1.0 - np.exp(-b * np.maximum(0.0 - c, 0.0)))
    )

def fit_model(vol_cm3, force_n, model_name):
    x = np.asarray(vol_cm3, float)
    y = np.asarray(force_n, float)

    if model_name == "linear0":
        # closed-form: a = (x·y)/(x·x)
        denom = float(np.sum(x * x))
        if denom <= 1e-18:
            return None
        a = float(np.sum(x * y) / denom)
        yhat = _m_linear0(x, a)
        return {
            "type": "linear0",
            "params": {"a": a},
            "equation": f"F = {a:.6g} * V",
            "yhat": yhat,
        }

    if model_name == "linear":
        A = np.column_stack([x, np.ones_like(x)])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        yhat = _m_linear(x, a, b)
        return {
            "type": "linear",
            "params": {"a": a, "b": b},
            "equation": f"F = {a:.6g} * V + {b:.6g}",
            "yhat": yhat,
        }

    if model_name == "poly2":
        if len(x) < 3:
            return None
        c2, c1, c0 = [float(v) for v in np.polyfit(x, y, deg=2)]
        yhat = _m_poly2(x, c2, c1, c0)
        return {
            "type": "poly2",
            "params": {"c2": c2, "c1": c1, "c0": c0},
            "equation": f"F = {c2:.6g} * V^2 + {c1:.6g} * V + {c0:.6g}",
            "yhat": yhat,
        }

    if model_name == "sat_exp":
        # a>=0, b>=0
        p0 = [max(np.max(y), 1e-6), 1.0]
        bounds = ([0.0, 0.0], [np.inf, np.inf])
        try:
            popt, _ = curve_fit(_m_sat_exp, x, y, p0=p0, bounds=bounds, maxfev=200000)
        except Exception:
            return None
        a, b = float(popt[0]), float(popt[1])
        yhat = _m_sat_exp(x, a, b)
        return {
            "type": "sat_exp",
            "params": {"a": a, "b": b},
            "equation": f"F = {a:.6g} * (1 - exp(-{b:.6g} * V))",
            "yhat": yhat,
        }

    if model_name == "growth":
        # a>=0, b>=0
        p0 = [max(np.max(y), 1e-6), 1.0]
        bounds = ([0.0, 0.0], [np.inf, np.inf])
        try:
            popt, _ = curve_fit(_m_growth, x, y, p0=p0, bounds=bounds, maxfev=200000)
        except Exception:
            return None
        a, b = float(popt[0]), float(popt[1])
        yhat = _m_growth(x, a, b)
        return {
            "type": "growth",
            "params": {"a": a, "b": b},
            "equation": f"F = {a:.6g} * (exp({b:.6g} * V) - 1)",
            "yhat": yhat,
        }

    if model_name == "hinge_saturating":
        # a>=0, b>=0, c free (allow a small negative c)
        xmax = float(np.max(x)) if len(x) else 1.0
        p0 = [max(np.max(y), 1e-6), 5.0, 0.1 * xmax]
        bounds = ([0.0, 0.0, -0.5 * xmax], [np.inf, np.inf, 1.5 * xmax])
        try:
            popt, _ = curve_fit(_m_hinge_sat, x, y, p0=p0, bounds=bounds, maxfev=400000)
        except Exception:
            return None
        a, b, c = float(popt[0]), float(popt[1]), float(popt[2])
        yhat = _m_hinge_sat(x, a, b, c)
        eq = (
            f"F = {a:.6g} * ( (1-exp(-{b:.6g}*max(V-{c:.6g},0)))"
            f" - (1-exp(-{b:.6g}*max(0-{c:.6g},0))) )"
        )
        return {
            "type": "hinge_saturating",
            "params": {"a": a, "b": b, "c": c},
            "equation": eq,
            "yhat": yhat,
        }

    return None

def predict(model, v):
    t = model["type"]
    p = model["params"]
    v = np.asarray(v, float)
    if t == "linear0":
        return _m_linear0(v, float(p["a"]))
    if t == "linear":
        return _m_linear(v, float(p["a"]), float(p["b"]))
    if t == "poly2":
        return _m_poly2(v, float(p["c2"]), float(p["c1"]), float(p["c0"]))
    if t == "sat_exp":
        return _m_sat_exp(v, float(p["a"]), float(p["b"]))
    if t == "growth":
        return _m_growth(v, float(p["a"]), float(p["b"]))
    if t == "hinge_saturating":
        return _m_hinge_sat(v, float(p["a"]), float(p["b"]), float(p["c"]))
    raise ValueError(f"Unknown model type: {t}")

def fit_best_model(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    candidates = []
    for name in MODEL_CANDIDATES:
        m = fit_model(x, y, name)
        if m is None:
            continue
        m["sse"] = float(np.sum((y - m["yhat"]) ** 2))
        m["rmse"] = rmse(y, m["yhat"])
        m["r2"] = r2_score(y, m["yhat"])
        candidates.append(m)

    if not candidates:
        raise RuntimeError("No model could be fit (check your data).")

    best = min(candidates, key=lambda d: d["rmse"])
    summary = [{"type": c["type"], "rmse": float(c["rmse"]), "r2": float(c["r2"]), "sse": float(c["sse"])}
               for c in sorted(candidates, key=lambda d: d["rmse"])]

    return best, summary


# ===========================
# VOLUME COMPUTATION
# ===========================
def depth_map_to_volume_cm3(height_map_mm, roi_mask, mm_per_px, depth_eps_mm=0.01):
    """
    Integrate V = sum(depth_mm * pixel_area_mm2) over depth>eps within ROI.
    Returns:
      volume_cm3, area_mm2, max_depth_mm
    """
    Z = np.asarray(height_map_mm, dtype=np.float32).copy()
    roi = roi_mask.astype(bool)

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
    return volume_cm3, area_mm2, max_depth_mm


# ===========================
# MAIN
# ===========================
def main():
    ensure_dir(OUTPUT_DIR)

    RUNS_DIR = os.path.join(OUTPUT_DIR, "ftp_runs")
    ensure_dir(RUNS_DIR)

    n_levels = len(FORCE_LEVELS_N)
    n_expected = n_levels * IMAGES_PER_LEVEL

    # Incremental output files
    csv_path = os.path.join(OUTPUT_DIR, "per_image_results.csv")
    jsonl_path = os.path.join(OUTPUT_DIR, "per_image_results.jsonl") 

    fieldnames = [
        "file",
        "force_N",
        "volume_cm3",
        "contact_area_mm2",
        "max_depth_mm",
        "mm_per_px",
        "estimated_grating_period_px",
        "ftp_output_dir",
    ]

    rows = []
    processed = set()
    csv_mode = "w"
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
                if "file" in row:
                    processed.add(row["file"])
        csv_mode = "a"

    with open(csv_path, csv_mode, newline="", encoding="utf-8") as fcsv, \
         open(jsonl_path, "a", encoding="utf-8") as fjsonl:

        w = csv.DictWriter(fcsv, fieldnames=fieldnames)
        if csv_mode == "w":
            w.writeheader()
            fcsv.flush()

        for i in range(n_expected):
            img_index = IMAGE_START_INDEX + i
            force = float(FORCE_LEVELS_N[i // IMAGES_PER_LEVEL])

            fname = IMAGE_PATTERN.format(img_index)
            path = os.path.join(DEFORMED_DIR, fname)

            if fname in processed:
                print(f"[SKIP] Already processed: {fname}")
                continue

            if not os.path.isfile(path):
                print(f"[SKIP] Missing: {path}")
                continue

            base = os.path.splitext(fname)[0] 
            img_out_dir = os.path.join(RUNS_DIR, f"{img_index:03d}_{base}_F{force:g}N")
            ensure_dir(img_out_dir)

            try:
                shutil.copy2(path, os.path.join(img_out_dir, fname))
            except Exception:
                pass

            res = shape_ftp.main(
                reference_path=REFERENCE_PATH,
                deformed_path=path,
                output_dir=img_out_dir,
                calibration_json=shape_ftp.CALIBRATION_JSON,
                batch_mode=True,
                save_summary_figures=True,
                export_heightmaps=False,   
                debug=False,
                return_results=True
            )

            height_mm = res["height_map_mm_crop"]
            roi = res["roi_eroded_crop"]
            est_period_px = res.get("estimated_grating_period_px", None)

            if OVERRIDE_MM_PER_PX is not None:
                mm_per_px = float(OVERRIDE_MM_PER_PX)
            else:
                if est_period_px is None or not np.isfinite(est_period_px) or est_period_px <= 1e-9:
                    raise RuntimeError(
                        f"{fname}: invalid estimated_grating_period_px. "
                        "Fix shape_ftp return or set OVERRIDE_MM_PER_PX."
                    )
                mm_per_px = float(GRATING_PITCH_MM) / float(est_period_px)

            vol_cm3, area_mm2, max_depth_mm = depth_map_to_volume_cm3(
                height_map_mm=height_mm,
                roi_mask=roi,
                mm_per_px=mm_per_px,
                depth_eps_mm=DEPTH_EPS_MM
            )

            row = {
                "file": fname,
                "force_N": force,
                "volume_cm3": vol_cm3,
                "contact_area_mm2": area_mm2,
                "max_depth_mm": max_depth_mm,
                "mm_per_px": mm_per_px,
                "estimated_grating_period_px": safe_float(est_period_px, np.nan),
                "ftp_output_dir": img_out_dir,
            }

            # Incremental save (CSV + JSONL) after each image
            w.writerow(row)
            fcsv.flush()

            fjsonl.write(json.dumps(row) + "\n")
            fjsonl.flush()

            rows.append(row)
            processed.add(fname)

            print(
                f"[OK] {fname} | F={force:.3g} N | V={vol_cm3:.6g} cm^3 | "
                f"area={area_mm2:.3g} mm^2 | maxDepth={max_depth_mm:.3g} mm | "
                f"mm/px={mm_per_px:.6g} | out={img_out_dir}"
            )

    if len(rows) < 10:
        raise RuntimeError("Not enough samples processed (check paths / filenames).")

    # --- Fit model ---
    V = np.array([float(r["volume_cm3"]) for r in rows], dtype=float)
    F = np.array([float(r["force_N"]) for r in rows], dtype=float)

    if ANCHOR_ORIGIN:
        V_fit = np.concatenate([np.zeros(int(ORIGIN_WEIGHT), float), V])
        F_fit = np.concatenate([np.zeros(int(ORIGIN_WEIGHT), float), F])
    else:
        V_fit, F_fit = V, F

    best, summary = fit_best_model(V_fit, F_fit)

    model_out = {
        "reference_path": REFERENCE_PATH,
        "deformed_dir": DEFORMED_DIR,
        "output_dir": OUTPUT_DIR,
        "volume_definition": f"V_cm3 = sum(depth_mm * (mm_per_px^2)) / 1000 over depth>{DEPTH_EPS_MM}mm in ROI",
        "grating_pitch_mm": float(GRATING_PITCH_MM),
        "depth_eps_mm": float(DEPTH_EPS_MM),
        "anchor_origin": bool(ANCHOR_ORIGIN),
        "origin_weight": int(ORIGIN_WEIGHT),
        "best_model": {
            "type": best["type"],
            "params": best["params"],
            "equation": best["equation"],
            "rmse": float(best["rmse"]),
            "r2": float(best["r2"]),
            "sse": float(best["sse"]),
            "n_fit": int(len(V_fit)),
            "n_samples": int(len(V)),
        },
        "candidates_summary": summary,
    }

    json_path = os.path.join(OUTPUT_DIR, "calibration_model.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(model_out, f, indent=2)

    # ---------------------------
    # Plots:
    # ---------------------------
    forces_sorted = FORCE_LEVELS_N[:] 
    vols_by_force = []
    for ff in forces_sorted:
        vols = [float(r["volume_cm3"]) for r in rows if abs(float(r["force_N"]) - ff) < 1e-12]
        vols_by_force.append(vols)

    # (1) Boxplot: Volume distribution per force
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.boxplot(vols_by_force, showfliers=True)
    ax1.set_xticks(range(1, len(forces_sorted) + 1))
    ax1.set_xticklabels([str(f) for f in forces_sorted], rotation=0)
    ax1.set_xlabel("Force (N)")
    ax1.set_ylabel("Integrated volume (cm³)")
    ax1.set_title("Indentation Volume Across Force Levels (5 images per force)")

    # jittered points
    rng = np.random.default_rng(0)
    for k, vols in enumerate(vols_by_force, start=1):
        if len(vols) == 0:
            continue
        xj = k + (rng.random(len(vols)) - 0.5) * 0.18
        ax1.scatter(xj, vols, s=18)

    fig1.tight_layout()
    boxplot_path = os.path.join(OUTPUT_DIR, "volume_by_force_boxplot.png")
    fig1.savefig(boxplot_path, dpi=200)
    plt.close(fig1)

    # (2) Scatter + fitted curve: Force vs Volume
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.scatter(V, F, s=18)

    vmin, vmax = float(np.min(V)), float(np.max(V))
    xs = np.linspace(max(0.0, vmin * 0.95), vmax * 1.05, 400)
    ys = predict(best, xs)
    ax2.plot(xs, ys)

    ax2.set_xlabel("Integrated volume (cm³)")
    ax2.set_ylabel("Force (N)")
    ax2.set_title("Force-Volume Calibration Curve")

    fig2.tight_layout()
    fitplot_path = os.path.join(OUTPUT_DIR, "force_vs_volume_fit.png")
    fig2.savefig(fitplot_path, dpi=200)
    plt.close(fig2)

    print("\n=== OUTPUTS ===")
    print(f"CSV (incremental): {csv_path}")
    print(f"JSONL (incremental): {jsonl_path}")
    print(f"JSON (final model): {json_path}")
    print(f"PLOT (boxplot): {boxplot_path}")
    print(f"PLOT (fit): {fitplot_path}")
    print("\n=== BEST MODEL ===")
    print(best["equation"])
    print(f"RMSE={best['rmse']:.6g} N | R2={best['r2']:.6g}")

    if best["type"] == "linear0":
        print(f"k (paper-style) = {best['params']['a']:.6g} N/cm^3")

if __name__ == "__main__":
    main()

