# temperature_color_model.py
#
# Robust FULL-COLOR TLC temperature calibration (Heating + Cooling + Global)


import glob
import os
import re
import json
import csv

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.isotonic import IsotonicRegression

try:
    import joblib
except Exception:
    joblib = None


# ===========================
# PLOT STYLE 
# ===========================
FIG_W = 12
FIG_H = FIG_W/2.8  

AXIS_LABEL_FONTSIZE = 18
TICK_LABEL_FONTSIZE = 16
TITLE_FONTSIZE = 20
LEGEND_FONTSIZE = 14

SCATTER_S_FRAMES = 30
SCATTER_S_TRUEPRED_FRAMES = 32
SCATTER_S_PER_TEMP_MEAN = 140
SCATTER_S_ROI_POINTS = 60

LINEWIDTH_TREND = 4.0
LINEWIDTH_TREND_HYST = 4.2
LINEWIDTH_TREND_GLOBAL = 4.5
LINEWIDTH_IDEAL = 3.0

ERRORBAR_MARKERSIZE = 9
ERRORBAR_ELINEWIDTH = 2.0
ERRORBAR_CAPSIZE = 4
ERRORBAR_CAPTHICK = 2.0

plt.rcParams.update({
    "axes.titlesize": TITLE_FONTSIZE,
    "axes.labelsize": AXIS_LABEL_FONTSIZE,
    "xtick.labelsize": TICK_LABEL_FONTSIZE,
    "ytick.labelsize": TICK_LABEL_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
})


# ===========================
# USER CONFIGURATION
# ===========================
HEATING_PATTERN = "./Temperature/Heating_Colored_Temp/heating-*.jpg"
COOLING_PATTERN = "./Temperature/Cooling_Colored_Temp/cooling_colored-*.jpg"

TEMPS_HEATING = list(range(20, 40)) + list(range(40, 76, 5))
TEMPS_COOLING = TEMPS_HEATING[::-1]
FRAMES_PER_TEMP = 5

# LAB only
USE_FEATURES = ("L", "a", "b") 

# Robust model settings
AUTO_SELECT_DEGREE = True
POLY_DEGREE = 2
POLY_DEGREE_CANDIDATES = [1, 2, 3, 4]
CV_SPLITS = 6

# Huber parameters
HUBER_EPSILON = 1.2
HUBER_ALPHA = 1e-6
HUBER_MAX_ITER = 10000

FIT_TEMP_RANGE = (20, 33)  

USE_ISOTONIC_CALIBRATION = True

# ---- OUTPUT ROOT ----
OUTPUT_ROOT_DIR = "./Temperature/Colored_Model"
CALIBRATION_FOLDER_NAME = "calibration_out"
OUT_DIR = os.path.join(OUTPUT_ROOT_DIR, CALIBRATION_FOLDER_NAME)

# Plot colors
COLOR_HEAT = "#fe8920"
COLOR_COOL = "#1f77b5"
COLOR_IDEAL = "#d72729"
COLOR_MEAN_GLOBAL = "#d72729"

# Forward-trend smoothing for plotting
FORWARD_TREND_STAT = "median"   
FORWARD_SMOOTH_WINDOW = 3      
FORWARD_INTERP_POINTS = 400


# ===========================
# CONSTANT ANNULUS ROI CONFIG
# ===========================
INNER_CIRCLE_P1 = (1881, 1749)
INNER_CIRCLE_P2 = (1579, 665)
INNER_CIRCLE_P3 = (2616, 936)

OUTER_CIRCLE_P1 = (1803, 1990)
OUTER_CIRCLE_P2 = (1393, 496)
OUTER_CIRCLE_P3 = (2856, 860)


# ===========================
# HELPERS
# ===========================
def sort_key_by_index(path: str):
    base = os.path.basename(path)
    match = re.search(r"-(\d+)\.", base)
    return int(match.group(1)) if match else base

def load_images_sorted(pattern: str):
    files = glob.glob(pattern)
    if not files:
        raise RuntimeError(f"No files found for pattern: {pattern}")
    files.sort(key=sort_key_by_index)
    return files

def circle_from_three_points(p1, p2, p3, eps=1e-12):
    x1, y1 = map(float, p1)
    x2, y2 = map(float, p2)
    x3, y3 = map(float, p3)

    a = x1 - x2
    b = y1 - y2
    c = x1 - x3
    d = y1 - y3

    e = (x1**2 - x2**2 + y1**2 - y2**2) / 2.0
    f = (x1**2 - x3**2 + y1**2 - y3**2) / 2.0

    det = a * d - b * c
    if abs(det) < eps:
        raise RuntimeError("Cannot define circle: points are collinear (or nearly collinear).")

    cx = (d * e - b * f) / det
    cy = (-c * e + a * f) / det
    r = np.hypot(x1 - cx, y1 - cy)
    return cx, cy, float(r)

def create_annulus_mask(h, w, inner_p1, inner_p2, inner_p3, outer_p1, outer_p2, outer_p3):
    cx_in, cy_in, r_in = circle_from_three_points(inner_p1, inner_p2, inner_p3)
    cx_out, cy_out, r_out = circle_from_three_points(outer_p1, outer_p2, outer_p3)
    if r_out <= r_in:
        raise RuntimeError("Invalid annulus: outer radius must be larger than inner radius.")

    Y, X = np.ogrid[:h, :w]
    dist_sq_in = (X - cx_in) ** 2 + (Y - cy_in) ** 2
    dist_sq_out = (X - cx_out) ** 2 + (Y - cy_out) ** 2

    mask_inner = dist_sq_in <= (r_in ** 2)
    mask_outer = dist_sq_out <= (r_out ** 2)
    mask = mask_outer & (~mask_inner)
    return mask, (cx_in, cy_in, r_in), (cx_out, cy_out, r_out)

def save_annulus_roi_overlay(image_bgr, mask, inner_circle, outer_circle, out_path):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    cx_in, cy_in, r_in = inner_circle
    cx_out, cy_out, r_out = outer_circle

    theta = np.linspace(0, 2 * np.pi, 500)
    x_in = cx_in + r_in * np.cos(theta)
    y_in = cy_in + r_in * np.sin(theta)
    x_out = cx_out + r_out * np.cos(theta)
    y_out = cy_out + r_out * np.sin(theta)

    plt.figure(figsize=(FIG_W, FIG_H))
    plt.imshow(image_rgb)
    plt.imshow(mask.astype(float), alpha=0.35)
    plt.plot(x_in, y_in, linewidth=2.5)
    plt.plot(x_out, y_out, linewidth=2.5)

    xs = [
        INNER_CIRCLE_P1[0], INNER_CIRCLE_P2[0], INNER_CIRCLE_P3[0],
        OUTER_CIRCLE_P1[0], OUTER_CIRCLE_P2[0], OUTER_CIRCLE_P3[0]
    ]
    ys = [
        INNER_CIRCLE_P1[1], INNER_CIRCLE_P2[1], INNER_CIRCLE_P3[1],
        OUTER_CIRCLE_P1[1], OUTER_CIRCLE_P2[1], OUTER_CIRCLE_P3[1]
    ]
    plt.scatter(xs, ys, s=SCATTER_S_ROI_POINTS)
    plt.title("Annulus ROI overlay (outer circle minus inner circle)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def compute_lab_features_for_sequence(image_files, mask):
    feats = []
    h0, w0 = None, None
    for path in image_files:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read image: {path}")

        h, w = img.shape[:2]
        if h0 is None:
            h0, w0 = h, w
        elif (h, w) != (h0, w0):
            raise RuntimeError(f"Image size mismatch for {path}")

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]
        a = lab[:, :, 1]
        b = lab[:, :, 2]

        feats.append([float(L[mask].mean()), float(a[mask].mean()), float(b[mask].mean())])
    return np.array(feats, dtype=float)

def group_by_temperature_features(features, temps, frames_per_temp):
    n_temps = len(temps)
    expected = n_temps * frames_per_temp
    if features.shape[0] != expected:
        raise RuntimeError(
            f"Images ({features.shape[0]}) != {n_temps} temps × {frames_per_temp} frames = {expected}"
        )
    blocks = []
    for i in range(n_temps):
        s = i * frames_per_temp
        e = s + frames_per_temp
        blocks.append(features[s:e, :])
    return np.array(temps, dtype=float), blocks

def compute_mean_std_features(features_per_temp):
    means = np.array([blk.mean(axis=0) for blk in features_per_temp], dtype=float)
    stds  = np.array([blk.std(axis=0, ddof=1) for blk in features_per_temp], dtype=float)
    return means, stds

def select_feature_matrix(X_full, use_features):
    name_to_idx = {"L": 0, "a": 1, "b": 2}
    idxs = [name_to_idx[n] for n in use_features]
    return X_full[:, idxs], idxs


# ===========================
# METRICS
# ===========================
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    abs_err = np.abs(y_true - y_pred)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    maxe = float(abs_err.max()) if abs_err.size else float("nan")
    p95 = float(np.percentile(abs_err, 95)) if abs_err.size else float("nan")

    return {
        "n": int(len(y_true)),
        "rmse_C": rmse,
        "mae_C": mae,
        "r2": r2,
        "max_abs_err_C": maxe,
        "p95_abs_err_C": p95,
    }

def report_metrics(y_true, y_pred, label):
    m = compute_metrics(y_true, y_pred)
    print(f"\n{label}")
    print(f"  RMSE = {m['rmse_C']:.3f} °C")
    print(f"  MAE  = {m['mae_C']:.3f} °C")
    print(f"  R^2  = {m['r2']:.4f}")
    print(f"  Max |err| = {m['max_abs_err_C']:.3f} °C")
    print(f"  95th pct |err| = {m['p95_abs_err_C']:.3f} °C")

def get_feature_names_from_poly(poly, input_feature_names):
    try:
        return poly.get_feature_names_out(input_feature_names)
    except Exception:
        return poly.get_feature_names(input_feature_names)

def polynomial_equation_string(pipeline_model, input_feature_names, precision=8):
    poly = pipeline_model.named_steps.get("polynomialfeatures", None)
    hub = pipeline_model.named_steps.get("huberregressor", None)
    if poly is None or hub is None:
        raise RuntimeError("Pipeline must include PolynomialFeatures and HuberRegressor.")

    terms = get_feature_names_from_poly(poly, list(input_feature_names))
    coefs = np.asarray(hub.coef_).ravel()
    intercept = float(hub.intercept_) if np.isscalar(hub.intercept_) else float(np.asarray(hub.intercept_).ravel()[0])

    parts = []
    if abs(intercept) > 1e-12:
        parts.append(f"{intercept:.{precision}g}")

    for c, term in zip(coefs, terms):
        if abs(c) < 1e-12:
            continue
        parts.append(f"({c:.{precision}g})*{term.replace(' ', '*')}")

    if not parts:
        return "T = 0  (all coefficients ~0)"
    eq = " + ".join(parts).replace("+ -", "- ")

    wrapped, cur = [], ""
    for tok in eq.split(" + "):
        if len(cur) + len(tok) + 3 > 120:
            if cur:
                wrapped.append(cur)
            cur = tok
        else:
            cur = tok if not cur else cur + " + " + tok
    if cur:
        wrapped.append(cur)

    return "T =\n  " + "\n  ".join(wrapped)


# ===========================
# MODEL BUILDING + DEGREE SELECTION
# ===========================
def make_huber_poly_model(degree: int):
    return make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        PolynomialFeatures(degree=int(degree), include_bias=True),
        HuberRegressor(
            epsilon=float(HUBER_EPSILON),
            alpha=float(HUBER_ALPHA),
            max_iter=int(HUBER_MAX_ITER),
        ),
    )

def choose_degree_by_groupcv(X, y, groups, degrees):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    groups = np.asarray(groups)

    uniq = np.unique(groups)
    n_splits = int(min(CV_SPLITS, len(uniq)))
    if n_splits < 2:
        return int(degrees[0])

    gkf = GroupKFold(n_splits=n_splits)
    best_deg, best_rmse = None, None

    for deg in degrees:
        rmses = []
        for tr, te in gkf.split(X, y, groups=groups):
            m = make_huber_poly_model(deg)
            m.fit(X[tr], y[tr])
            pred = m.predict(X[te])
            rmses.append(float(np.sqrt(mean_squared_error(y[te], pred))))
        mean_rmse = float(np.mean(rmses))
        if (best_rmse is None) or (mean_rmse < best_rmse):
            best_rmse = mean_rmse
            best_deg = int(deg)

    return int(best_deg)


# ===========================
# FORWARD TREND FOR PLOTTING 
# ===========================
def _moving_average(y, window: int):
    y = np.asarray(y, float)
    if window <= 1 or y.size < 3:
        return y
    window = int(window)
    if window % 2 == 0:
        window += 1
    if y.size < window:
        return y
    k = np.ones(window, float) / float(window)
    y_pad = np.pad(y, (window // 2, window // 2), mode="edge")
    return np.convolve(y_pad, k, mode="valid")

def forward_trend_from_true_temperature(T_true_frames, feat_frames,
                                       stat="median",
                                       smooth_window=3,
                                       n_interp=400):
    T_true_frames = np.asarray(T_true_frames, float)
    feat_frames = np.asarray(feat_frames, float)

    uniq_T = np.array(sorted(np.unique(T_true_frames)), dtype=float)
    y_stat = []
    for t in uniq_T:
        sel = (T_true_frames == t)
        vals = feat_frames[sel]
        y_stat.append(float(np.median(vals)) if stat != "mean" else float(np.mean(vals)))
    y_stat = np.asarray(y_stat, float)

    ok = np.isfinite(uniq_T) & np.isfinite(y_stat)
    uniq_T = uniq_T[ok]
    y_stat = y_stat[ok]
    if uniq_T.size < 2:
        return np.array([]), np.array([])

    y_smooth = _moving_average(y_stat, smooth_window)
    x_grid = np.linspace(float(np.min(uniq_T)), float(np.max(uniq_T)), int(n_interp))
    y_grid = np.interp(x_grid, uniq_T, y_smooth)
    return x_grid, y_grid


# ===========================
# PREDICTION CALIBRATION
# ===========================
def fit_isotonic_calibrator(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(y_pred, y_true)
    return ir

def apply_calibrator(calibrator, y_pred):
    if calibrator is None:
        return np.asarray(y_pred, float)
    return calibrator.predict(np.asarray(y_pred, float))


# ===========================
# PER-TEMPERATURE MEAN OF FRAME PREDICTIONS
# ===========================
def per_temp_mean_pred(y_true_frames, y_pred_frames):
    y_true_frames = np.asarray(y_true_frames, float)
    y_pred_frames = np.asarray(y_pred_frames, float)
    uniq_T = np.array(sorted(np.unique(y_true_frames)), dtype=float)

    pred_means = []
    for t in uniq_T:
        sel = (y_true_frames == t)
        pred_means.append(float(np.mean(y_pred_frames[sel])) if np.any(sel) else float("nan"))
    return uniq_T, np.asarray(pred_means, float)


# ===========================
# PLOTTING
# ===========================
def plot_L_vs_T_with_forward_trend(out_path, title, color_run,
                                  temps_true_frames, L_frames,
                                  temps_true_means, L_means, L_stds):
    plt.figure(figsize=(FIG_W, FIG_H))
    rng = np.random.default_rng(0)
    jitter = (rng.random(len(L_frames)) - 0.5) * 0.2
    plt.scatter(np.asarray(temps_true_frames, float) + jitter,
                np.asarray(L_frames, float),
                alpha=0.25, s=SCATTER_S_FRAMES, color=color_run, label="Measured frames")
    plt.errorbar(np.asarray(temps_true_means, float),
                 np.asarray(L_means, float),
                 yerr=np.asarray(L_stds, float),
                 fmt="o", capsize=ERRORBAR_CAPSIZE, color=color_run,
                 markersize=ERRORBAR_MARKERSIZE,
                 elinewidth=ERRORBAR_ELINEWIDTH,
                 capthick=ERRORBAR_CAPTHICK,
                 label="Measured mean ± std")

    x_curve, y_curve = forward_trend_from_true_temperature(
        temps_true_frames, L_frames,
        stat=FORWARD_TREND_STAT,
        smooth_window=FORWARD_SMOOTH_WINDOW,
        n_interp=FORWARD_INTERP_POINTS
    )
    if x_curve.size:
        plt.plot(x_curve, y_curve, color=color_run, linewidth=LINEWIDTH_TREND, label="Trend (from measured frames)")

    plt.xlabel("Temperature [°C]")
    plt.ylabel("Mean LAB L in ROI")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_hysteresis_L_vs_T_models(out_path,
                                 temps_heat_means, L_heat_means, L_heat_stds, y_heat_frames, L_heat_frames,
                                 temps_cool_means, L_cool_means, L_cool_stds, y_cool_frames, L_cool_frames):
    plt.figure(figsize=(FIG_W, FIG_H))
    plt.errorbar(temps_heat_means, L_heat_means, yerr=L_heat_stds, fmt="o",
                 capsize=ERRORBAR_CAPSIZE, color=COLOR_HEAT,
                 markersize=ERRORBAR_MARKERSIZE,
                 elinewidth=ERRORBAR_ELINEWIDTH,
                 capthick=ERRORBAR_CAPTHICK,
                 label="Heating (mean ± std)")
    plt.errorbar(temps_cool_means, L_cool_means, yerr=L_cool_stds, fmt="s",
                 capsize=ERRORBAR_CAPSIZE, color=COLOR_COOL,
                 markersize=ERRORBAR_MARKERSIZE,
                 elinewidth=ERRORBAR_ELINEWIDTH,
                 capthick=ERRORBAR_CAPTHICK,
                 label="Cooling (mean ± std)")

    xh, yh = forward_trend_from_true_temperature(y_heat_frames, L_heat_frames,
                                                 stat=FORWARD_TREND_STAT,
                                                 smooth_window=FORWARD_SMOOTH_WINDOW,
                                                 n_interp=FORWARD_INTERP_POINTS)
    xc, yc = forward_trend_from_true_temperature(y_cool_frames, L_cool_frames,
                                                 stat=FORWARD_TREND_STAT,
                                                 smooth_window=FORWARD_SMOOTH_WINDOW,
                                                 n_interp=FORWARD_INTERP_POINTS)
    if xh.size:
        plt.plot(xh, yh, linewidth=LINEWIDTH_TREND_HYST, color=COLOR_HEAT, label="Heating trend")
    if xc.size:
        plt.plot(xc, yc, linewidth=LINEWIDTH_TREND_HYST, color=COLOR_COOL, label="Cooling trend")

    plt.xlabel("Temperature [°C]")
    plt.ylabel("Mean LAB L in ROI")
    plt.title("Color TLC Hysteresis Model (Heating vs Cooling) — LAB L vs Temperature")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_global_true_vs_pred_split(out_path,
                                  y_heat_true, y_heat_pred,
                                  y_cool_true, y_cool_pred,
                                  y_means_true=None, y_means_pred=None):
    y_heat_true = np.asarray(y_heat_true, float)
    y_heat_pred = np.asarray(y_heat_pred, float)
    y_cool_true = np.asarray(y_cool_true, float)
    y_cool_pred = np.asarray(y_cool_pred, float)

    all_true = [y_heat_true, y_cool_true]
    all_pred = [y_heat_pred, y_cool_pred]
    if y_means_true is not None and y_means_pred is not None:
        all_true.append(np.asarray(y_means_true, float))
        all_pred.append(np.asarray(y_means_pred, float))

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    tmin = float(min(np.min(all_true), np.min(all_pred)))
    tmax = float(max(np.max(all_true), np.max(all_pred)))

    plt.figure(figsize=(FIG_W, FIG_H))
    plt.plot([tmin, tmax], [tmin, tmax], linestyle="--", linewidth=LINEWIDTH_IDEAL,
             color=COLOR_IDEAL, label="Ideal (y=x)")
    plt.scatter(y_heat_true, y_heat_pred, s=SCATTER_S_TRUEPRED_FRAMES, alpha=0.22,
                color=COLOR_HEAT, label="Heating frames")
    plt.scatter(y_cool_true, y_cool_pred, s=SCATTER_S_TRUEPRED_FRAMES, alpha=0.22,
                color=COLOR_COOL, label="Cooling frames")
    if y_means_true is not None and y_means_pred is not None:
        plt.scatter(np.asarray(y_means_true, float), np.asarray(y_means_pred, float),
                    s=SCATTER_S_PER_TEMP_MEAN, alpha=0.95,
                    color=COLOR_IDEAL, label="Per-temp mean prediction")

    plt.xlabel("Measured Temperature [°C]")
    plt.ylabel("Predicted Temperature [°C]")
    plt.title("Color TLC Global Model — Measured vs Predicted Temperature")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_global_L_vs_T_all(out_path,
                          y_heat_frames, L_heat_frames,
                          y_cool_frames, L_cool_frames,
                          uniq_T, L_all_means, L_all_stds):
    plt.figure(figsize=(FIG_W, FIG_H))
    rng = np.random.default_rng(0)

    plt.scatter(y_heat_frames + (rng.random(len(y_heat_frames)) - 0.5) * 0.2, L_heat_frames,
                alpha=0.18, s=SCATTER_S_FRAMES, color=COLOR_HEAT, label="Measured frames (heating)")
    plt.scatter(y_cool_frames + (rng.random(len(y_cool_frames)) - 0.5) * 0.2, L_cool_frames,
                alpha=0.18, s=SCATTER_S_FRAMES, color=COLOR_COOL, label="Measured frames (cooling)")

    plt.errorbar(uniq_T, L_all_means, yerr=L_all_stds, fmt="o", capsize=ERRORBAR_CAPSIZE,
                 color=COLOR_MEAN_GLOBAL,
                 markersize=ERRORBAR_MARKERSIZE,
                 elinewidth=ERRORBAR_ELINEWIDTH,
                 capthick=ERRORBAR_CAPTHICK,
                 label="Measured mean ± std (all data)")

    y_all_frames = np.concatenate([y_heat_frames, y_cool_frames])
    L_all_frames = np.concatenate([L_heat_frames, L_cool_frames])
    xg, yg = forward_trend_from_true_temperature(
        y_all_frames, L_all_frames,
        stat=FORWARD_TREND_STAT,
        smooth_window=FORWARD_SMOOTH_WINDOW,
        n_interp=FORWARD_INTERP_POINTS
    )
    if xg.size:
        plt.plot(xg, yg, linewidth=LINEWIDTH_TREND_GLOBAL, color=COLOR_IDEAL, label="Trend (from measured frames)")

    plt.xlabel("Temperature [°C]")
    plt.ylabel("Mean LAB L in ROI")
    plt.title("Color TLC Global Model — LAB L vs T")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_per_temp_error_csv(out_csv, y_true_frames, y_pred_frames, label):
    y_true_frames = np.asarray(y_true_frames, float)
    y_pred_frames = np.asarray(y_pred_frames, float)
    uniq_T = np.array(sorted(np.unique(y_true_frames)), dtype=float)

    rows = []
    for t in uniq_T:
        sel = (y_true_frames == t)
        err = y_pred_frames[sel] - t
        rows.append({
            "label": label,
            "T_true": float(t),
            "n_frames": int(np.count_nonzero(sel)),
            "mean_pred": float(np.mean(y_pred_frames[sel])),
            "mean_err": float(np.mean(err)),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "max_abs_err": float(np.max(np.abs(err))),
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def plot_per_temp_mae(out_path, y_true_frames, y_pred_frames, title):
    y_true_frames = np.asarray(y_true_frames, float)
    y_pred_frames = np.asarray(y_pred_frames, float)
    uniq_T = np.array(sorted(np.unique(y_true_frames)), dtype=float)
    maes = []
    for t in uniq_T:
        sel = (y_true_frames == t)
        maes.append(float(np.mean(np.abs(y_pred_frames[sel] - t))))
    plt.figure(figsize=(FIG_W, FIG_H))
    plt.plot(uniq_T, maes, marker="o", linewidth=LINEWIDTH_TREND, markersize=ERRORBAR_MARKERSIZE)
    plt.xlabel("Temperature [°C]")
    plt.ylabel("Mean Absolute Error [°C]")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ===========================
# MAIN
# ===========================
def main():
    np.random.seed(0)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Load heating + ROI ----
    heating_files = load_images_sorted(HEATING_PATTERN)
    print(f"Found {len(heating_files)} HEATING images.")

    first_img = cv2.imread(heating_files[0], cv2.IMREAD_COLOR)
    if first_img is None:
        raise RuntimeError(f"Could not read first heating image: {heating_files[0]}")

    h, w = first_img.shape[:2]
    mask, inner_circle, outer_circle = create_annulus_mask(
        h, w,
        INNER_CIRCLE_P1, INNER_CIRCLE_P2, INNER_CIRCLE_P3,
        OUTER_CIRCLE_P1, OUTER_CIRCLE_P2, OUTER_CIRCLE_P3
    )

    overlay_path = os.path.join(OUT_DIR, "00_roi_overlay.png")
    save_annulus_roi_overlay(first_img, mask, inner_circle, outer_circle, overlay_path)

    # ---- Heating features ----
    heating_feats = compute_lab_features_for_sequence(heating_files, mask)
    temps_heat, blocks_heat = group_by_temperature_features(heating_feats, TEMPS_HEATING, FRAMES_PER_TEMP)
    means_heat, stds_heat = compute_mean_std_features(blocks_heat)

    # ---- Cooling features ----
    cooling_files = load_images_sorted(COOLING_PATTERN)
    print(f"Found {len(cooling_files)} COOLING images.")

    test_img = cv2.imread(cooling_files[0], cv2.IMREAD_COLOR)
    if test_img is None:
        raise RuntimeError(f"Could not read first cooling image: {cooling_files[0]}")
    if test_img.shape[:2] != (h, w):
        raise RuntimeError("Cooling image size mismatch vs heating images; ROI would be invalid.")

    cooling_feats = compute_lab_features_for_sequence(cooling_files, mask)
    temps_cool, blocks_cool = group_by_temperature_features(cooling_feats, TEMPS_COOLING, FRAMES_PER_TEMP)
    means_cool, stds_cool = compute_mean_std_features(blocks_cool)

    # ---- Frame-level true temps ----
    y_heat_frames_full = np.repeat(temps_heat, FRAMES_PER_TEMP)
    y_cool_frames_full = np.repeat(temps_cool, FRAMES_PER_TEMP)

    # ---- Regression matrices ----
    X_heat_frames_full, _ = select_feature_matrix(heating_feats, USE_FEATURES)
    X_cool_frames_full, _ = select_feature_matrix(cooling_feats, USE_FEATURES)

    def apply_range(X, y):
        if FIT_TEMP_RANGE is None:
            return X, y
        t0, t1 = float(FIT_TEMP_RANGE[0]), float(FIT_TEMP_RANGE[1])
        sel = (y >= t0) & (y <= t1)
        return X[sel], y[sel]

    X_heat_frames, y_heat_frames = apply_range(X_heat_frames_full, y_heat_frames_full)
    X_cool_frames, y_cool_frames = apply_range(X_cool_frames_full, y_cool_frames_full)

    print("\nFitting range:",
          "FULL" if FIT_TEMP_RANGE is None else f"[{FIT_TEMP_RANGE[0]}, {FIT_TEMP_RANGE[1]}] °C")
    print(f"  Heating frames used: {len(y_heat_frames)} / {len(y_heat_frames_full)}")
    print(f"  Cooling frames used: {len(y_cool_frames)} / {len(y_cool_frames_full)}")

    g_heat = y_heat_frames.astype(int)
    g_cool = y_cool_frames.astype(int)

    # ===========================
    # HEATING MODEL
    # ===========================
    deg_h = choose_degree_by_groupcv(X_heat_frames, y_heat_frames, g_heat, POLY_DEGREE_CANDIDATES) if AUTO_SELECT_DEGREE else int(POLY_DEGREE)
    model_heat = make_huber_poly_model(deg_h)
    model_heat.fit(X_heat_frames, y_heat_frames)
    pred_heat_raw = model_heat.predict(X_heat_frames)

    heat_cal = fit_isotonic_calibrator(y_heat_frames, pred_heat_raw) if USE_ISOTONIC_CALIBRATION else None
    pred_heat = apply_calibrator(heat_cal, pred_heat_raw)

    y_heat_means_true, pred_heat_means = per_temp_mean_pred(y_heat_frames, pred_heat)

    report_metrics(y_heat_frames, pred_heat, f"Heating FINAL model (frames): deg={deg_h}" + (" + isotonic" if USE_ISOTONIC_CALIBRATION else ""))
    report_metrics(y_heat_means_true, pred_heat_means, f"Heating FINAL model (per-temp mean pred): deg={deg_h}" + (" + isotonic" if USE_ISOTONIC_CALIBRATION else ""))

    # ===========================
    # COOLING MODEL
    # ===========================
    deg_c = choose_degree_by_groupcv(X_cool_frames, y_cool_frames, g_cool, POLY_DEGREE_CANDIDATES) if AUTO_SELECT_DEGREE else int(POLY_DEGREE)
    model_cool = make_huber_poly_model(deg_c)
    model_cool.fit(X_cool_frames, y_cool_frames)
    pred_cool_raw = model_cool.predict(X_cool_frames)

    cool_cal = fit_isotonic_calibrator(y_cool_frames, pred_cool_raw) if USE_ISOTONIC_CALIBRATION else None
    pred_cool = apply_calibrator(cool_cal, pred_cool_raw)

    y_cool_means_true, pred_cool_means = per_temp_mean_pred(y_cool_frames, pred_cool)

    report_metrics(y_cool_frames, pred_cool, f"Cooling FINAL model (frames): deg={deg_c}" + (" + isotonic" if USE_ISOTONIC_CALIBRATION else ""))
    report_metrics(y_cool_means_true, pred_cool_means, f"Cooling FINAL model (per-temp mean pred): deg={deg_c}" + (" + isotonic" if USE_ISOTONIC_CALIBRATION else ""))

    # ===========================
    # GLOBAL MODEL
    # ===========================
    X_all_frames = np.vstack([X_heat_frames, X_cool_frames])
    y_all_frames = np.concatenate([y_heat_frames, y_cool_frames])
    g_all = y_all_frames.astype(int)

    deg_g = choose_degree_by_groupcv(X_all_frames, y_all_frames, g_all, POLY_DEGREE_CANDIDATES) if AUTO_SELECT_DEGREE else int(POLY_DEGREE)
    model_global = make_huber_poly_model(deg_g)
    model_global.fit(X_all_frames, y_all_frames)
    pred_all_raw = model_global.predict(X_all_frames)

    glob_cal = fit_isotonic_calibrator(y_all_frames, pred_all_raw) if USE_ISOTONIC_CALIBRATION else None
    pred_all = apply_calibrator(glob_cal, pred_all_raw)

    n_heat = len(y_heat_frames)
    y_heat_pred_global = pred_all[:n_heat]
    y_cool_pred_global = pred_all[n_heat:]

    y_all_means_true, pred_all_means = per_temp_mean_pred(y_all_frames, pred_all)

    report_metrics(y_all_frames, pred_all, f"GLOBAL FINAL model (frames): deg={deg_g}" + (" + isotonic" if USE_ISOTONIC_CALIBRATION else ""))
    report_metrics(y_all_means_true, pred_all_means, f"GLOBAL FINAL model (per-temp mean pred): deg={deg_g}" + (" + isotonic" if USE_ISOTONIC_CALIBRATION else ""))

    # ===========================
    # Save equations + metrics JSON 
    # ===========================
    eq_heat = polynomial_equation_string(model_heat, USE_FEATURES, precision=8)
    eq_cool = polynomial_equation_string(model_cool, USE_FEATURES, precision=8)
    eq_glob = polynomial_equation_string(model_global, USE_FEATURES, precision=8)

    eq_path = os.path.join(OUT_DIR, "equations_color_models_final.txt")
    with open(eq_path, "w", encoding="utf-8") as f:
        f.write("FINAL MODEL (HEATING)  T = f(L,a,b)\n")
        f.write(f"Degree={deg_h}\n{eq_heat}\n\n")
        f.write("FINAL MODEL (COOLING)  T = f(L,a,b)\n")
        f.write(f"Degree={deg_c}\n{eq_cool}\n\n")
        f.write("FINAL MODEL (GLOBAL / MERGED)  T = f(L,a,b)\n")
        f.write(f"Degree={deg_g}\n{eq_glob}\n")

    summary = {
        "output_dir": os.path.abspath(OUT_DIR),
        "use_features": list(USE_FEATURES),
        "frames_per_temp": int(FRAMES_PER_TEMP),
        "poly_degree_candidates": list(POLY_DEGREE_CANDIDATES),
        "auto_select_degree": bool(AUTO_SELECT_DEGREE),
        "fit_temp_range": None if FIT_TEMP_RANGE is None else [float(FIT_TEMP_RANGE[0]), float(FIT_TEMP_RANGE[1])],
        "use_isotonic_calibration": bool(USE_ISOTONIC_CALIBRATION),
        "huber": {
            "epsilon": float(HUBER_EPSILON),
            "alpha": float(HUBER_ALPHA),
            "max_iter": int(HUBER_MAX_ITER),
        },
        "models_final": {
            "heating": {
                "degree": int(deg_h),
                "equation": eq_heat,
                "metrics_frames": compute_metrics(y_heat_frames, pred_heat),
                "metrics_means": compute_metrics(y_heat_means_true, pred_heat_means),
            },
            "cooling": {
                "degree": int(deg_c),
                "equation": eq_cool,
                "metrics_frames": compute_metrics(y_cool_frames, pred_cool),
                "metrics_means": compute_metrics(y_cool_means_true, pred_cool_means),
            },
            "global": {
                "degree": int(deg_g),
                "equation": eq_glob,
                "metrics_frames": compute_metrics(y_all_frames, pred_all),
                "metrics_means": compute_metrics(y_all_means_true, pred_all_means),
            },
        },
        "plot_forward_trend": {
            "stat": FORWARD_TREND_STAT,
            "smooth_window": int(FORWARD_SMOOTH_WINDOW),
            "interp_points": int(FORWARD_INTERP_POINTS),
        },
    }

    metrics_json_path = os.path.join(OUT_DIR, "models_final_summary_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    L_heat_frames_full = heating_feats[:, 0]
    L_cool_frames_full = cooling_feats[:, 0]
    L_heat_means = means_heat[:, 0]
    L_cool_means = means_cool[:, 0]
    L_heat_stds = stds_heat[:, 0]
    L_cool_stds = stds_cool[:, 0]

    # ===========================
    # PLOTS
    # ===========================
    plot_L_vs_T_with_forward_trend(
        out_path=os.path.join(OUT_DIR, "01_heating_L_vs_T_modelcurve.png"),
        title="Heating: LAB L vs Temperature (measured points + trend curve)",
        color_run=COLOR_HEAT,
        temps_true_frames=y_heat_frames_full,
        L_frames=L_heat_frames_full,
        temps_true_means=temps_heat,
        L_means=L_heat_means,
        L_stds=L_heat_stds,
    )

    plot_L_vs_T_with_forward_trend(
        out_path=os.path.join(OUT_DIR, "02_cooling_L_vs_T_modelcurve.png"),
        title="Cooling: LAB L vs Temperature (measured points + trend curve)",
        color_run=COLOR_COOL,
        temps_true_frames=y_cool_frames_full,
        L_frames=L_cool_frames_full,
        temps_true_means=temps_cool,
        L_means=L_cool_means,
        L_stds=L_cool_stds,
    )

    plot_hysteresis_L_vs_T_models(
        out_path=os.path.join(OUT_DIR, "03_hysteresis_L_vs_T_models.png"),
        temps_heat_means=temps_heat, L_heat_means=L_heat_means, L_heat_stds=L_heat_stds,
        y_heat_frames=y_heat_frames_full, L_heat_frames=L_heat_frames_full,
        temps_cool_means=temps_cool, L_cool_means=L_cool_means, L_cool_stds=L_cool_stds,
        y_cool_frames=y_cool_frames_full, L_cool_frames=L_cool_frames_full,
    )

    plot_global_true_vs_pred_split(
        out_path=os.path.join(OUT_DIR, "05_global_true_vs_pred.png"),
        y_heat_true=y_heat_frames,
        y_heat_pred=y_heat_pred_global,
        y_cool_true=y_cool_frames,
        y_cool_pred=y_cool_pred_global,
        y_means_true=y_all_means_true,
        y_means_pred=pred_all_means,
    )

    # Global L mean±std 
    y_all_frames_full = np.concatenate([y_heat_frames_full, y_cool_frames_full])
    L_all_frames_full = np.concatenate([L_heat_frames_full, L_cool_frames_full])
    uniq_T_full = np.array(sorted(np.unique(y_all_frames_full)), dtype=float)
    L_all_means = []
    L_all_stds = []
    for t in uniq_T_full:
        sel = (y_all_frames_full == t)
        L_all_means.append(float(np.mean(L_all_frames_full[sel])))
        L_all_stds.append(float(np.std(L_all_frames_full[sel], ddof=1)) if np.count_nonzero(sel) > 1 else 0.0)

    plot_global_L_vs_T_all(
        out_path=os.path.join(OUT_DIR, "06_global_L_vs_T_all_trend.png"),
        y_heat_frames=y_heat_frames_full, L_heat_frames=L_heat_frames_full,
        y_cool_frames=y_cool_frames_full, L_cool_frames=L_cool_frames_full,
        uniq_T=uniq_T_full, L_all_means=np.asarray(L_all_means, float), L_all_stds=np.asarray(L_all_stds, float),
    )

    # ===========================
    # Error vs temperature
    # ===========================
    save_per_temp_error_csv(os.path.join(OUT_DIR, "07_global_per_temp_error.csv"),
                            y_all_frames, pred_all, label="global")
    plot_per_temp_mae(os.path.join(OUT_DIR, "08_global_mae_vs_T.png"),
                      y_all_frames, pred_all,
                      title="Global model MAE vs Temperature" + (" (fit-range applied)" if FIT_TEMP_RANGE is not None else ""))

    # ===========================
    # Save models 
    # ===========================
    if joblib is not None:
        out_heat = os.path.join(OUT_DIR, f"color_model_heating_huber_deg{deg_h}.joblib")
        out_cool = os.path.join(OUT_DIR, f"color_model_cooling_huber_deg{deg_c}.joblib")
        out_glob = os.path.join(OUT_DIR, f"color_model_global_huber_deg{deg_g}.joblib")

        joblib.dump(
            {
                "model": model_heat,
                "use_features": USE_FEATURES,
                "poly_degree": deg_h,
                "regressor": "HuberRegressor",
                "scaler": "StandardScaler",
                "isotonic_calibrator": heat_cal,
                "fit_temp_range": FIT_TEMP_RANGE,
            },
            out_heat
        )
        joblib.dump(
            {
                "model": model_cool,
                "use_features": USE_FEATURES,
                "poly_degree": deg_c,
                "regressor": "HuberRegressor",
                "scaler": "StandardScaler",
                "isotonic_calibrator": cool_cal,
                "fit_temp_range": FIT_TEMP_RANGE,
            },
            out_cool
        )
        joblib.dump(
            {
                "model": model_global,
                "use_features": USE_FEATURES,
                "poly_degree": deg_g,
                "regressor": "HuberRegressor",
                "scaler": "StandardScaler",
                "isotonic_calibrator": glob_cal,
                "fit_temp_range": FIT_TEMP_RANGE,
            },
            out_glob
        )

    print("\nSaved:")
    print(f"  Output folder: {os.path.abspath(OUT_DIR)}")
    print(f"  ROI overlay:   {overlay_path}")
    print(f"  Metrics JSON:  {metrics_json_path}")
    print(f"  Equations:     {eq_path}")
    print(f"  Diagnostics:   07_global_per_temp_error.csv, 08_global_mae_vs_T.png")
    if joblib is not None:
        print("  Models saved (joblib).")


if __name__ == "__main__":
    main()
