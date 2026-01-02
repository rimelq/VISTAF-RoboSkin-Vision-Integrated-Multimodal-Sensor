# multimodal_sensor.py
#
# Unified multimodal tactile sensor: Force + Temperature + Shape
#
# Centralizes all outputs from:
#   - force_sensor.py (FTP height -> volume -> force)
#   - temperature_sensor.py (color + black TLC models -> temperature map)
#   - All calibration submodels (phase_to_height, height_to_force, temperature models)
#
# Outputs:
#   - Combined sensor readings (force, temperature stats, shape metrics)
#   - All debug/final visualizations from both modules
#   - Complete performance metrics from all calibration models
#   - 3D heightmap visualization (optional interactive window)
#   - JSON summary


from __future__ import annotations

import os
import json
import shutil
import glob
from typing import Dict, Any, Optional
from datetime import datetime

import numpy as np
import matplotlib

# ============================================================
# USER CONFIG - PATHS
# ============================================================

REFERENCE_IMAGE = "./Final_demos_images/FINAL_reference.jpg"
DEFORMED_IMAGE  = "./Final_demos_images/FINAL_E_deformed.jpg"
OUTPUT_ROOT     = "./Multimodal_Sensor/run_output"

# Interactive 3D heightmap window 
SHOW_3D_HEIGHTMAP_INTERACTIVE = True

if SHOW_3D_HEIGHTMAP_INTERACTIVE:
    for b in ("QtAgg", "Qt5Agg", "TkAgg"):
        try:
            matplotlib.use(b, force=True)
            break
        except Exception:
            pass
else:
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import force_sensor
import temperature_sensor


# ============================================================
# CALIBRATION MODEL PATHS (for loading metrics)
# ============================================================

# Phase-to-height calibration
PHASE_TO_HEIGHT_JSON = "./Force/Phase_to_height/calibration_out/calibration_model.json"

# Height-to-force calibration
HEIGHT_TO_FORCE_JSON = "./Force/Height_to_force/calibration_out/calibration_model.json"

# Temperature color model (LAB)
TEMP_COLOR_METRICS_JSON = "./Temperature/Colored_Model/calibration_out/models_final_summary_metrics.json"

# Temperature black model (LAB + gray)
TEMP_BLACK_METRICS_JSON = "./Temperature/MixedColorBlack_Model/calibration_out/models_final_summary_metrics.json"


# ============================================================
# INTERNAL PATHS (auto-generated)
# ============================================================

def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

TIMESTAMP = get_timestamp()
SESSION_DIR = os.path.join(OUTPUT_ROOT, f"session_{TIMESTAMP}")

FORCE_SUBDIR = os.path.join(SESSION_DIR, "force_sensing")
TEMP_SUBDIR = os.path.join(SESSION_DIR, "temperature_sensing")
COMBINED_SUBDIR = os.path.join(SESSION_DIR, "combined_outputs")


# ============================================================
# UTIL
# ============================================================

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

def load_json_safe(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"[WARN] Calibration JSON not found: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {e}")
        return None

def find_file_recursive(root_dir: str, filename: str) -> Optional[str]:
    matches = glob.glob(os.path.join(root_dir, "**", filename), recursive=True)
    if not matches:
        return None
    # prefer newest if multiple
    return max(matches, key=os.path.getmtime)


def save_force_shape_heightmap_right_panel(
    ftp_out_dir: str,
    combined_out_dir: str,
    force_N: float,
    src_filename: str = "07_phase_and_height_FINAL_SMOOTH_ROI.png",
) -> Optional[str]:
    """
    Takes the RIGHT part of `src_filename` (assumed side-by-side panel image),
    REMOVES the embedded old title by cropping it away,
    and re-adds a NEW title using matplotlib default styling.
    """
    import cv2  # local import

    src = find_file_recursive(ftp_out_dir, src_filename)
    if src is None or not os.path.exists(src):
        print(f"[WARN] Could not find {src_filename} under: {ftp_out_dir}")
        return None

    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Failed to read image: {src}")
        return None

    h, w = img.shape[:2]
    if w < 4:
        print(f"[WARN] Unexpected width for {src}: {w}")
        return None

    # Crop right half (panel)
    x0 = w // 2
    x0 = min(max(x0 + 2, 0), w)
    right = img[:, x0:w].copy()

    # Crop away the embedded old title area at the top of the PNG
    # Tune this fraction if needed (0.18..0.25 are typical)
    y_cut = int(0.07 * right.shape[0])
    right_no_title = right[y_cut:, :, :]

    # Render with matplotlib so the title uses default matplotlib style
    title = f"Deformation Heightmap (mm) - Force: {force_N:.3f} N"
    rgb = cv2.cvtColor(right_no_title, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(6.0, 6.0), dpi=200)
    ax = fig.add_subplot(111)
    ax.imshow(rgb)
    ax.set_axis_off()
    ax.set_title(title)  # default matplotlib title style

    dst = os.path.join(combined_out_dir, "force_shape_heightmap.png")
    fig.savefig(dst, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return dst


# ============================================================
# 3D HEIGHTMAP VISUALIZATION (optional interactive window)
# ============================================================

def plot_height_map_interactive(height_map, circ_mask=None, title="Height map (interactive 3D)"):
    h, w = height_map.shape
    Y, X = np.mgrid[0:h, 0:w]
    Z = height_map.astype(float).copy()

    if circ_mask is not None:
        if circ_mask.shape != height_map.shape:
            print(f"[WARN] circ_mask shape {circ_mask.shape} != height_map shape {height_map.shape}. "
                  f"Falling back to finite-mask of heightmap.")
            circ_mask = np.isfinite(height_map)
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


# ============================================================
# METRIC EXTRACTION HELPERS
# ============================================================

def extract_phase_to_height_metrics(calib: Optional[Dict]) -> Dict[str, Any]:
    if calib is None:
        return {}

    best = calib.get("best_model", {})
    return {
        "calibration_type": "phase_to_height",
        "model_type": best.get("type", "unknown"),
        "equation": best.get("equation", ""),
        "r2": safe_float(best.get("r2", np.nan)),
        "rmse": safe_float(best.get("rmse", np.nan)),
        "n_samples": int(best.get("n", 0)),
        "x_definition": calib.get("x_definition", ""),
    }

def extract_height_to_force_metrics(calib: Optional[Dict]) -> Dict[str, Any]:
    if calib is None:
        return {}

    best = calib.get("best_model", {})
    return {
        "calibration_type": "height_to_force",
        "model_type": best.get("type", "unknown"),
        "equation": best.get("equation", ""),
        "r2": safe_float(best.get("r2", np.nan)),
        "rmse": safe_float(best.get("rmse", np.nan)),
        "n_fit": int(best.get("n_fit", 0)),
        "n_samples": int(best.get("n_samples", 0)),
        "volume_definition": calib.get("volume_definition", ""),
    }

def extract_temp_model_metrics(calib: Optional[Dict], model_name: str) -> Dict[str, Any]:
    """Extract metrics for heating/cooling/global from temperature calibration JSON"""
    if calib is None:
        return {}

    models = calib.get("models_final", {})
    if model_name not in models:
        return {}

    m = models[model_name]

    frames = m.get("metrics_frames", {})
    means = m.get("metrics_means", {})

    return {
        "model": model_name,
        "degree": int(m.get("degree", 0)),
        "equation": m.get("equation", ""),
        "frames": {
            "rmse_C": safe_float(frames.get("rmse_C", np.nan)),
            "mae_C": safe_float(frames.get("mae_C", np.nan)),
            "r2": safe_float(frames.get("r2", np.nan)),
            "max_abs_err_C": safe_float(frames.get("max_abs_err_C", np.nan)),
            "p95_abs_err_C": safe_float(frames.get("p95_abs_err_C", np.nan)),
            "n": int(frames.get("n", 0)),
        },
        "means": {
            "rmse_C": safe_float(means.get("rmse_C", np.nan)),
            "mae_C": safe_float(means.get("mae_C", np.nan)),
            "r2": safe_float(means.get("r2", np.nan)),
            "max_abs_err_C": safe_float(means.get("max_abs_err_C", np.nan)),
            "p95_abs_err_C": safe_float(means.get("p95_abs_err_C", np.nan)),
            "n": int(means.get("n", 0)),
        },
    }


# ============================================================
# PRINT METRICS TO CONSOLE 
# ============================================================

def print_section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def print_calibration_metrics():
    """
    Terminal output: ONLY
      - TLC colored: global
      - TLC black:   global
    (Everything else remains available in the JSON summary.)
    """
    print_section("CALIBRATION MODEL PERFORMANCE METRICS (GLOBAL TLC ONLY)")

    # Temperature Color Model - GLOBAL only
    color_calib = load_json_safe(TEMP_COLOR_METRICS_JSON)
    if color_calib:
        metrics = extract_temp_model_metrics(color_calib, "global")
        if metrics:
            print("\n--- TLC COLORED (LAB) GLOBAL ---")
            print(f"Degree: {metrics['degree']}")
            frames = metrics.get("frames", {})
            means = metrics.get("means", {})
            print("Frame-level:")
            print(f"  RMSE = {frames.get('rmse_C', np.nan):.3f} °C")
            print(f"  MAE  = {frames.get('mae_C', np.nan):.3f} °C")
            print(f"  R²   = {frames.get('r2', np.nan):.4f}")
            print(f"  Max |err| = {frames.get('max_abs_err_C', np.nan):.3f} °C")
            print(f"  95th pct |err| = {frames.get('p95_abs_err_C', np.nan):.3f} °C")
            print("Per-temp means:")
            print(f"  RMSE = {means.get('rmse_C', np.nan):.3f} °C")
            print(f"  MAE  = {means.get('mae_C', np.nan):.3f} °C")
            print(f"  R²   = {means.get('r2', np.nan):.4f}")
            print(f"  Max |err| = {means.get('max_abs_err_C', np.nan):.3f} °C")
            print(f"  95th pct |err| = {means.get('p95_abs_err_C', np.nan):.3f} °C")

    # Temperature Black Model - GLOBAL only
    black_calib = load_json_safe(TEMP_BLACK_METRICS_JSON)
    if black_calib:
        metrics = extract_temp_model_metrics(black_calib, "global")
        if metrics:
            print("\n--- TLC BLACK (LAB+GRAY) GLOBAL ---")
            print(f"Degree: {metrics['degree']}")
            frames = metrics.get("frames", {})
            means = metrics.get("means", {})
            print("Frame-level:")
            print(f"  RMSE = {frames.get('rmse_C', np.nan):.3f} °C")
            print(f"  MAE  = {frames.get('mae_C', np.nan):.3f} °C")
            print(f"  R²   = {frames.get('r2', np.nan):.4f}")
            print(f"  Max |err| = {frames.get('max_abs_err_C', np.nan):.3f} °C")
            print(f"  95th pct |err| = {frames.get('p95_abs_err_C', np.nan):.3f} °C")
            print("Per-temp means:")
            print(f"  RMSE = {means.get('rmse_C', np.nan):.3f} °C")
            print(f"  MAE  = {means.get('mae_C', np.nan):.3f} °C")
            print(f"  R²   = {means.get('r2', np.nan):.4f}")
            print(f"  Max |err| = {means.get('max_abs_err_C', np.nan):.3f} °C")
            print(f"  95th pct |err| = {means.get('p95_abs_err_C', np.nan):.3f} °C")


# ============================================================
# MAIN MULTIMODAL SENSING
# ============================================================

def main():
    ensure_dir(SESSION_DIR)
    ensure_dir(FORCE_SUBDIR)
    ensure_dir(TEMP_SUBDIR)
    ensure_dir(COMBINED_SUBDIR)

    print_section(f"MULTIMODAL TACTILE SENSOR - Session {TIMESTAMP}")
    print(f"Reference: {REFERENCE_IMAGE}")
    print(f"Deformed:  {DEFORMED_IMAGE}")
    print(f"Output:    {SESSION_DIR}")

    # ========================================
    # FORCE SENSING (with shape)
    # ========================================
    print_section("FORCE & SHAPE SENSING")

    # Create output directory
    ftp_out_dir = os.path.join(FORCE_SUBDIR, "ftp_run")
    ensure_dir(ftp_out_dir)

    # Call shape_ftp directly to get the results (including ROI)
    import shape_ftp

    shape_results = shape_ftp.main(
        reference_path=REFERENCE_IMAGE,
        deformed_path=DEFORMED_IMAGE,
        output_dir=ftp_out_dir,
        calibration_json=getattr(shape_ftp, "CALIBRATION_JSON", None),
        batch_mode=True,
        save_summary_figures=True,
        export_heightmaps=True,
        debug=False,
        return_results=True,
    )

    # Extract height map and ROI from shape_ftp results
    height_mm = shape_results["height_map_mm_crop"]

    # Create ROI from finite values in heightmap (most reliable method)
    force_roi_mask = np.isfinite(height_mm)

    est_period_px = shape_results.get("estimated_grating_period_px", None)

    # Now compute force using the same logic as force_sensor.py
    from force_sensor import (
        load_force_calibration,
        predict_force_from_volume,
        depth_map_to_volume_cm3,
        FORCE_CALIBRATION_JSON,
        GRATING_PITCH_MM,
        DEPTH_EPS_MM,
    )

    force_calib = load_force_calibration(FORCE_CALIBRATION_JSON)
    best_model = force_calib["best_model"]

    # Estimate scale
    if est_period_px is not None and np.isfinite(est_period_px) and est_period_px > 1e-12:
        mm_per_px = float(GRATING_PITCH_MM) / float(est_period_px)
    else:
        raise RuntimeError("Invalid estimated_grating_period_px from shape_ftp")

    # Compute volume and force
    volume_cm3, contact_area_mm2, max_depth_mm = depth_map_to_volume_cm3(
        height_map_mm=height_mm,
        roi_mask=force_roi_mask,
        mm_per_px=mm_per_px,
        depth_eps_mm=DEPTH_EPS_MM,
    )

    force_N = predict_force_from_volume(best_model, volume_cm3)

    # Save results (same format as force_sensor.py)
    import csv
    force_results = {
        "reference_path": REFERENCE_IMAGE,
        "deformed_path": DEFORMED_IMAGE,
        "output_dir": FORCE_SUBDIR,
        "ftp_output_dir": ftp_out_dir,
        "grating_pitch_mm": float(GRATING_PITCH_MM),
        "depth_eps_mm": float(DEPTH_EPS_MM),
        "estimated_grating_period_px": float(est_period_px) if est_period_px else None,
        "mm_per_px": float(mm_per_px),
        "volume_cm3": float(volume_cm3),
        "contact_area_mm2": float(contact_area_mm2),
        "max_depth_mm": float(max_depth_mm),
        "force_N": float(force_N),
        "force_model": {
            "type": best_model.get("type", ""),
            "params": best_model.get("params", {}),
            "equation": best_model.get("equation", ""),
            "rmse": best_model.get("rmse", None),
            "r2": best_model.get("r2", None),
        },
    }

    json_path = os.path.join(FORCE_SUBDIR, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(force_results, f, indent=2)

    csv_path = os.path.join(FORCE_SUBDIR, "result.csv")
    fieldnames = ["reference_path", "deformed_path", "volume_cm3", "force_N",
                  "contact_area_mm2", "max_depth_mm", "mm_per_px",
                  "estimated_grating_period_px", "ftp_output_dir", "force_model_type"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow({
            "reference_path": REFERENCE_IMAGE,
            "deformed_path": DEFORMED_IMAGE,
            "volume_cm3": float(volume_cm3),
            "force_N": float(force_N),
            "contact_area_mm2": float(contact_area_mm2),
            "max_depth_mm": float(max_depth_mm),
            "mm_per_px": float(mm_per_px),
            "estimated_grating_period_px": float(est_period_px) if est_period_px else None,
            "ftp_output_dir": ftp_out_dir,
            "force_model_type": best_model.get("type", ""),
        })

    # Force summary prints (kept)
    print(f"\n[FORCE] {force_N:.3f} N")
    print(f"[VOLUME] {volume_cm3:.6g} cm³")
    print(f"[CONTACT AREA] {contact_area_mm2:.3f} mm²")
    print(f"[MAX DEPTH] {max_depth_mm:.3f} mm")
    print(f"[SCALE] {mm_per_px:.6g} mm/px")

    # Save requested force shape heightmap image into combined_outputs
    force_shape_img = save_force_shape_heightmap_right_panel(
        ftp_out_dir=ftp_out_dir,
        combined_out_dir=COMBINED_SUBDIR,
        force_N=force_N,
        src_filename="07_phase_and_height_FINAL_SMOOTH_ROI.png",
    )
    if force_shape_img:
        print(f"[FORCE] Saved force shape heightmap: {force_shape_img}")

    # ========================================
    # TEMPERATURE SENSING
    # ========================================
    print_section("TEMPERATURE SENSING")

    # Temporarily override temperature_sensor paths
    original_temp_in = temperature_sensor.INPUT_IMAGE_PATH
    original_temp_out = temperature_sensor.OUTPUT_DIR

    temperature_sensor.INPUT_IMAGE_PATH = DEFORMED_IMAGE
    temperature_sensor.OUTPUT_DIR = TEMP_SUBDIR

    # Run temperature sensor
    temperature_sensor.main()

    # Restore
    temperature_sensor.INPUT_IMAGE_PATH = original_temp_in
    temperature_sensor.OUTPUT_DIR = original_temp_out

    # Load temperature results
    temp_map_final = np.load(os.path.join(TEMP_SUBDIR, "temperature_map_final.npy"))
    th, tw = temp_map_final.shape

    # Compute temperature statistics
    roi_mask_path = os.path.join(TEMP_SUBDIR, "mask_roi.png")
    temp_roi_mask = None

    # If mask_roi.png is cropped, it won't match the full-size .npy. Recompute ROI geometry in that case.
    if os.path.exists(roi_mask_path):
        import cv2
        m = cv2.imread(roi_mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            m = m > 127
            if m.shape == temp_map_final.shape:
                temp_roi_mask = m
            else:
                print(f"[WARN] Temperature mask_roi.png shape {m.shape} != temperature map shape {temp_map_final.shape}. "
                      f"Recomputing ROI mask from temperature_sensor geometry.")

    if temp_roi_mask is None:
        try:
            roi_outer = temperature_sensor.roi_mask_from_circle(
                th, tw,
                temperature_sensor.OUTER_CIRCLE_P1,
                temperature_sensor.OUTER_CIRCLE_P2,
                temperature_sensor.OUTER_CIRCLE_P3,
            )
            if bool(getattr(temperature_sensor, "USE_INNER_CIRCLE", False)):
                roi_full = temperature_sensor.annulus_mask(
                    th, tw,
                    temperature_sensor.INNER_CIRCLE_P1,
                    temperature_sensor.INNER_CIRCLE_P2,
                    temperature_sensor.INNER_CIRCLE_P3,
                    temperature_sensor.OUTER_CIRCLE_P1,
                    temperature_sensor.OUTER_CIRCLE_P2,
                    temperature_sensor.OUTER_CIRCLE_P3,
                )
            else:
                roi_full = roi_outer

            # Match temperature_sensor behavior: when cropping toggle is enabled, stats are for OUTER ROI.
            if bool(getattr(temperature_sensor, "CROP_OUTPUT_TO_OUTER_ROI", False)):
                temp_roi_mask = roi_outer
            else:
                temp_roi_mask = roi_full
        except Exception as e:
            print(f"[WARN] Failed to reconstruct temperature ROI from geometry: {e}. Falling back to finite map mask.")
            temp_roi_mask = None

    if temp_roi_mask is not None:
        valid = temp_roi_mask & np.isfinite(temp_map_final)
    else:
        valid = np.isfinite(temp_map_final)

    if np.any(valid):
        temp_mean = float(np.mean(temp_map_final[valid]))
        temp_std = float(np.std(temp_map_final[valid]))
        temp_min = float(np.min(temp_map_final[valid]))
        temp_max = float(np.max(temp_map_final[valid]))
        temp_median = float(np.median(temp_map_final[valid]))
    else:
        temp_mean = temp_std = temp_min = temp_max = temp_median = float("nan")

    # Temperature summary prints (kept)
    print(f"\n[TEMPERATURE] Mean: {temp_mean:.2f} °C")
    print(f"[TEMPERATURE] Median: {temp_median:.2f} °C")
    print(f"[TEMPERATURE] Std: {temp_std:.2f} °C")
    print(f"[TEMPERATURE] Range: [{temp_min:.2f}, {temp_max:.2f}] °C")

    # ========================================
    # COLLECT CALIBRATION METRICS (TERMINAL: GLOBAL TLC ONLY)
    # ========================================
    print_calibration_metrics()

    # ========================================
    # COMBINED OUTPUT
    # ========================================
    print_section("GENERATING COMBINED OUTPUTS")

    # Load calibration JSONs (for JSON summary)
    p2h_calib = load_json_safe(PHASE_TO_HEIGHT_JSON)
    h2f_calib = load_json_safe(HEIGHT_TO_FORCE_JSON)
    color_calib = load_json_safe(TEMP_COLOR_METRICS_JSON)
    black_calib = load_json_safe(TEMP_BLACK_METRICS_JSON)

    # Build comprehensive summary
    summary = {
        "session_id": TIMESTAMP,
        "timestamp": datetime.now().isoformat(),
        "input_images": {
            "reference": REFERENCE_IMAGE,
            "deformed": DEFORMED_IMAGE,
        },
        "output_directory": SESSION_DIR,

        "sensor_readings": {
            "force": {
                "force_N": force_N,
                "volume_cm3": volume_cm3,
                "contact_area_mm2": contact_area_mm2,
                "max_depth_mm": max_depth_mm,
                "scale_mm_per_px": mm_per_px,
            },
            "temperature": {
                "mean_C": temp_mean,
                "median_C": temp_median,
                "std_C": temp_std,
                "min_C": temp_min,
                "max_C": temp_max,
                "valid_pixels": int(np.count_nonzero(valid)),
            },
        },

        "calibration_performance": {
            "phase_to_height": extract_phase_to_height_metrics(p2h_calib),
            "height_to_force": extract_height_to_force_metrics(h2f_calib),
            "temperature_color_model": {
                "heating": extract_temp_model_metrics(color_calib, "heating"),
                "cooling": extract_temp_model_metrics(color_calib, "cooling"),
                "global": extract_temp_model_metrics(color_calib, "global"),
            } if color_calib else {},
            "temperature_black_model": {
                "heating": extract_temp_model_metrics(black_calib, "heating"),
                "cooling": extract_temp_model_metrics(black_calib, "cooling"),
                "global": extract_temp_model_metrics(black_calib, "global"),
            } if black_calib else {},
        },

        "file_paths": {
            "force_subdir": FORCE_SUBDIR,
            "temperature_subdir": TEMP_SUBDIR,
            "combined_subdir": COMBINED_SUBDIR,
        },
    }

    # Save comprehensive JSON
    summary_json = os.path.join(COMBINED_SUBDIR, "multimodal_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved comprehensive summary: {summary_json}")

    # Copy key artifacts to combined folder
    print("\nCopying key artifacts to combined output...")

    # Force artifacts
    for fname in ["result.json", "result.csv"]:
        src = os.path.join(FORCE_SUBDIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(COMBINED_SUBDIR, f"force_{fname}"))

    # Temperature artifacts
    for fname in [
        "temperature_map_final_colormap.png",
        "temperature_map_final_colormap_overlay.png",
        "temperature_legend_horizontal.png",
    ]:
        src = os.path.join(TEMP_SUBDIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(COMBINED_SUBDIR, f"temp_{fname}"))

    # ========================================
    # OPTIONAL: 3D HEIGHTMAP VISUALIZATION (interactive window only)
    # ========================================
    if SHOW_3D_HEIGHTMAP_INTERACTIVE:
        print_section("3D HEIGHTMAP VISUALIZATION (INTERACTIVE WINDOW)")
        print("Opening interactive 3D window (close to continue)...")
        fig_interactive = plot_height_map_interactive(
            height_mm,
            circ_mask=force_roi_mask,
            title=f"3D Heightmap (Interactive) - Force: {force_N:.2f} N",
        )
        plt.show()
        plt.close(fig_interactive)

    # ========================================
    # FINAL SUMMARY
    # ========================================
    print_section("MULTIMODAL SENSING COMPLETE")
    print(f"\nSession directory: {SESSION_DIR}")
    print(f"\nForce sensing outputs: {FORCE_SUBDIR}")
    print(f"Temperature sensing outputs: {TEMP_SUBDIR}")
    print(f"Combined outputs: {COMBINED_SUBDIR}")

    print("\n--- SENSOR READINGS ---")
    print(f"Force:       {force_N:.3f} N")
    print(f"Temperature: {temp_mean:.2f} ± {temp_std:.2f} °C")
    print(f"Temperature min/max range: [{temp_min:.2f}, {temp_max:.2f}] °C")
    print(f"Max Depth:   {max_depth_mm:.3f} mm")
    print(f"Contact Area: {contact_area_mm2:.1f} mm²")

    print("\nAll outputs saved successfully!")


if __name__ == "__main__":
    main()

