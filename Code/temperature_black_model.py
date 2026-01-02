# temperature_black_model.py
#
# Robust Black TLC temperature calibration (Heating + Cooling + Global)


import glob
import os
import re
import json

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
HEATING_PATTERN = "./Temperature/Heating_MixedColorBlack_Temp/heating_mixed-*.jpg"
COOLING_PATTERN = "./Temperature/Cooling_MixedColorBlack_Temp/cooling_mixed-*.jpg"

TEMPS_HEATING = list(range(20, 51)) + list(range(55, 76, 5))
TEMPS_COOLING = TEMPS_HEATING[::-1]
FRAMES_PER_TEMP = 5

# Model features
USE_FEATURES = ("L", "a", "b", "gray") 

TRAIN_ON_PIXEL_SAMPLES = True
PIXELS_PER_IMAGE = 4000           
MAX_TOTAL_PIXEL_SAMPLES = 1_500_000 
RANDOM_SEED = 0

EXCLUDE_SATURATED_PIXELS = True
SAT_THRESH_GRAY = 245

# Robust model settings
AUTO_SELECT_DEGREE = True
POLY_DEGREE = 2  

# Reduced default candidates for stability 
POLY_DEGREE_CANDIDATES = [1, 2, 3]
CV_SPLITS = 6

# Huber parameters 
HUBER_EPSILON = 1.2
HUBER_ALPHA = 1e-4
HUBER_MAX_ITER = 10000

# Trend curve settings
TREND_BINS = 60
TREND_MIN_COUNT_PER_BIN = 2
TREND_STAT = "median" 

# Output directory
OUTPUT_ROOT_DIR = "./Temperature/MixedColorBlack_Model"
CALIBRATION_FOLDER_NAME = "calibration_out"
OUT_DIR = os.path.join(OUTPUT_ROOT_DIR, CALIBRATION_FOLDER_NAME)

COLOR_HEAT = "#fe8920"          # heating
COLOR_COOL = "#1f77b5"          # cooling
COLOR_ORANGE_HEX = "#d72729"
COLOR_STD = "#d72729"
COLOR_GLOBAL_TREND = "#d72729"


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
def sort_key_by_index(path):
    base = os.path.basename(path)
    match = re.search(r"-(\d+)\.", base)
    if match:
        return int(match.group(1))
    return base

def load_images_sorted(pattern):
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

def compute_mean_features_for_sequence(image_files, mask):
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
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        feats.append([
            float(L[mask].mean()),
            float(a[mask].mean()),
            float(b[mask].mean()),
            float(gray[mask].mean()),
        ])
    return np.array(feats, dtype=float)

def _sample_pixel_features_from_image(img_bgr, mask, rng, n_samples, exclude_saturated, sat_thresh):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    valid = mask.copy()
    if exclude_saturated:
        valid = valid & (gray < float(sat_thresh))

    coords = np.argwhere(valid)
    if coords.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    if coords.shape[0] <= n_samples:
        chosen = coords
    else:
        idx = rng.choice(coords.shape[0], size=int(n_samples), replace=False)
        chosen = coords[idx]

    yy = chosen[:, 0]
    xx = chosen[:, 1]

    L = lab[yy, xx, 0]
    a = lab[yy, xx, 1]
    b = lab[yy, xx, 2]
    g = gray[yy, xx]

    X = np.stack([L, a, b, g], axis=1).astype(np.float32)
    return X

def compute_pixel_samples_dataset(image_files, mask, y_true_frames,
                                  pixels_per_image, max_total_samples,
                                  seed=0, exclude_saturated=True, sat_thresh=245):
    rng = np.random.default_rng(int(seed))
    X_list = []
    y_list = []

    total = 0
    for i, path in enumerate(image_files):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read image: {path}")

        Xi = _sample_pixel_features_from_image(
            img, mask, rng,
            n_samples=int(pixels_per_image),
            exclude_saturated=bool(exclude_saturated),
            sat_thresh=int(sat_thresh),
        )
        if Xi.shape[0] == 0:
            continue

        yi = float(y_true_frames[i])
        X_list.append(Xi)
        y_list.append(np.full((Xi.shape[0],), yi, dtype=np.float32))
        total += Xi.shape[0]

        if total >= int(max_total_samples):
            break

    if not X_list:
        raise RuntimeError("No pixel samples collected. Check ROI mask / saturation threshold.")
    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.float32)
    return X, y

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
    name_to_idx = {"L": 0, "a": 1, "b": 2, "gray": 3}
    idxs = [name_to_idx[n] for n in use_features]
    return X_full[:, idxs], idxs


# ===========================
# METRICS + EQUATION STRING
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
        pretty = term.replace(" ", "*")
        parts.append(f"({c:.{precision}g})*{pretty}")

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
# ROBUST MODEL SELECTION
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
    n_groups = len(uniq)
    n_splits = int(min(CV_SPLITS, n_groups))
    if n_splits < 2:
        return int(degrees[0])

    gkf = GroupKFold(n_splits=n_splits)

    best_deg = None
    best_rmse = None

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
# MODEL TREND CURVE
# ===========================
def robust_trend_curve(T_pred, Y_feat, bins=60, min_count=2, stat="median"):
    x = np.asarray(T_pred, float)
    y = np.asarray(Y_feat, float)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 5:
        return np.array([]), np.array([])

    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax - xmin < 1e-9:
        return np.array([]), np.array([])

    edges = np.linspace(xmin, xmax, int(bins) + 1)
    x_curve = []
    y_curve = []

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sel = (x >= lo) & (x < hi) if i < len(edges) - 2 else (x >= lo) & (x <= hi)
        if np.count_nonzero(sel) < int(min_count):
            continue

        xb = 0.5 * (lo + hi)
        yb = float(np.median(y[sel])) if stat != "mean" else float(np.mean(y[sel]))
        x_curve.append(float(xb))
        y_curve.append(yb)

    x_curve = np.array(x_curve, float)
    y_curve = np.array(y_curve, float)
    order = np.argsort(x_curve)
    return x_curve[order], y_curve[order]


# ===========================
# PLOTS 
# ===========================
def plot_gray_vs_T_with_model(
    out_path,
    title,
    color_run,
    temps_true_frames,
    gray_frames,
    temps_true_means,
    gray_means,
    gray_stds,
    T_pred_frames,
    legend_model_label,
):
    plt.figure(figsize=(FIG_W, FIG_H))

    rng = np.random.default_rng(0)
    jitter = (rng.random(len(gray_frames)) - 0.5) * 0.2
    plt.scatter(temps_true_frames + jitter, gray_frames, alpha=0.25, s=SCATTER_S_FRAMES, color=color_run, label="Measured frames")

    plt.errorbar(
        temps_true_means, gray_means, yerr=gray_stds,
        fmt="o",
        capsize=ERRORBAR_CAPSIZE,
        color=color_run,
        markersize=ERRORBAR_MARKERSIZE,
        elinewidth=ERRORBAR_ELINEWIDTH,
        capthick=ERRORBAR_CAPTHICK,
        label="Measured mean ± std"
    )

    x_curve, y_curve = robust_trend_curve(
        T_pred_frames, gray_frames, bins=TREND_BINS, min_count=TREND_MIN_COUNT_PER_BIN, stat=TREND_STAT
    )
    if x_curve.size:
        plt.plot(x_curve, y_curve, color=color_run, linewidth=LINEWIDTH_TREND, label=legend_model_label)

    plt.xlabel("Temperature [°C]")
    plt.ylabel("Mean GRAY in ROI")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_hysteresis_gray_vs_T_models(
    out_path,
    temps_heat_means, g_heat_means, g_heat_stds, T_pred_heat_frames, g_heat_frames,
    temps_cool_means, g_cool_means, g_cool_stds, T_pred_cool_frames, g_cool_frames,
):
    plt.figure(figsize=(FIG_W, FIG_H))

    plt.errorbar(temps_heat_means, g_heat_means, yerr=g_heat_stds, fmt="o",
                 capsize=ERRORBAR_CAPSIZE, color=COLOR_HEAT,
                 markersize=ERRORBAR_MARKERSIZE,
                 elinewidth=ERRORBAR_ELINEWIDTH,
                 capthick=ERRORBAR_CAPTHICK,
                 label="Heating (measured mean ± std)")
    plt.errorbar(temps_cool_means, g_cool_means, yerr=g_cool_stds, fmt="s",
                 capsize=ERRORBAR_CAPSIZE, color=COLOR_COOL,
                 markersize=ERRORBAR_MARKERSIZE,
                 elinewidth=ERRORBAR_ELINEWIDTH,
                 capthick=ERRORBAR_CAPTHICK,
                 label="Cooling (measured mean ± std)")

    xh, yh = robust_trend_curve(T_pred_heat_frames, g_heat_frames, TREND_BINS, TREND_MIN_COUNT_PER_BIN, TREND_STAT)
    xc, yc = robust_trend_curve(T_pred_cool_frames, g_cool_frames, TREND_BINS, TREND_MIN_COUNT_PER_BIN, TREND_STAT)

    if xh.size:
        plt.plot(xh, yh, color=COLOR_HEAT, linewidth=LINEWIDTH_TREND_HYST, label="Heating model trend")
    if xc.size:
        plt.plot(xc, yc, color=COLOR_COOL, linewidth=LINEWIDTH_TREND_HYST, label="Cooling model trend")

    plt.xlabel("Temperature [°C]")
    plt.ylabel("Mean GRAY in ROI")
    plt.title("Black TLC Hysteresis Model (Heating vs Cooling) — Grayscale vs Temperature")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_global_gray_vs_T_all_modelcurve(
    out_path,
    y_heat_frames, g_heat_frames,
    y_cool_frames, g_cool_frames,
    uniq_T, g_all_means, g_all_stds,
    T_pred_all_frames, g_all_frames,
):
    plt.figure(figsize=(FIG_W, FIG_H))

    rng = np.random.default_rng(0)

    plt.scatter(
        y_heat_frames + (rng.random(len(y_heat_frames)) - 0.5) * 0.2, g_heat_frames,
        alpha=0.18, s=SCATTER_S_FRAMES, color=COLOR_HEAT, label="Measured frames (heating)"
    )
    plt.scatter(
        y_cool_frames + (rng.random(len(y_cool_frames)) - 0.5) * 0.2, g_cool_frames,
        alpha=0.18, s=SCATTER_S_FRAMES, color=COLOR_COOL, label="Measured frames (cooling)"
    )

    plt.errorbar(
        uniq_T, g_all_means, yerr=g_all_stds, fmt="o",
        capsize=ERRORBAR_CAPSIZE,
        color=COLOR_STD, ecolor=COLOR_STD,
        markersize=ERRORBAR_MARKERSIZE,
        elinewidth=ERRORBAR_ELINEWIDTH,
        capthick=ERRORBAR_CAPTHICK,
        label="Measured mean ± std (all data)"
    )

    xg, yg = robust_trend_curve(T_pred_all_frames, g_all_frames, TREND_BINS, TREND_MIN_COUNT_PER_BIN, TREND_STAT)
    if xg.size:
        plt.plot(xg, yg, linewidth=LINEWIDTH_TREND_GLOBAL, color=COLOR_GLOBAL_TREND, label="Global model trend")

    plt.xlabel("Temperature [°C]")
    plt.ylabel("Mean GRAY in ROI")
    plt.title("Black TLC Global Model — Grayscale vs Temperature")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_05_global_true_vs_pred(
    out_path,
    y_heat_true, y_heat_pred,
    y_cool_true, y_cool_pred,
    y_means_true, y_means_pred,
):
    y_heat_true = np.asarray(y_heat_true, float)
    y_heat_pred = np.asarray(y_heat_pred, float)
    y_cool_true = np.asarray(y_cool_true, float)
    y_cool_pred = np.asarray(y_cool_pred, float)

    all_true = np.concatenate([y_heat_true, y_cool_true, np.asarray(y_means_true, float)])
    all_pred = np.concatenate([y_heat_pred, y_cool_pred, np.asarray(y_means_pred, float)])

    tmin = float(min(np.min(all_true), np.min(all_pred)))
    tmax = float(max(np.max(all_true), np.max(all_pred)))

    plt.figure(figsize=(FIG_W, FIG_H))
    plt.plot([tmin, tmax], [tmin, tmax], linestyle="--", linewidth=LINEWIDTH_IDEAL,
             color=COLOR_ORANGE_HEX, label="Ideal (y=x)")

    plt.scatter(y_heat_true, y_heat_pred, s=SCATTER_S_TRUEPRED_FRAMES, alpha=0.25, color=COLOR_HEAT, label="Heating frames")
    plt.scatter(y_cool_true, y_cool_pred, s=SCATTER_S_TRUEPRED_FRAMES, alpha=0.25, color=COLOR_COOL, label="Cooling frames")

    plt.scatter(np.asarray(y_means_true, float), np.asarray(y_means_pred, float),
                s=SCATTER_S_PER_TEMP_MEAN, alpha=0.95, color=COLOR_ORANGE_HEX, label="Per-temp means (all data)")

    plt.xlabel("Measured Temperature [°C]")
    plt.ylabel("Predicted Temperature [°C]")
    plt.title("Black TLC Global Model — Measured vs Predicted Temperature")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ===========================
# MAIN
# ===========================
def main():
    np.random.seed(RANDOM_SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

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

    # ---- Per-image MEAN features (plots + degree selection + reporting) ----
    heating_feats_mean = compute_mean_features_for_sequence(heating_files, mask)
    temps_heat, blocks_heat = group_by_temperature_features(heating_feats_mean, TEMPS_HEATING, FRAMES_PER_TEMP)
    means_heat, stds_heat = compute_mean_std_features(blocks_heat)

    cooling_files = load_images_sorted(COOLING_PATTERN)
    print(f"Found {len(cooling_files)} COOLING images.")

    test_img = cv2.imread(cooling_files[0], cv2.IMREAD_COLOR)
    if test_img is None:
        raise RuntimeError(f"Could not read first cooling image: {cooling_files[0]}")
    if test_img.shape[:2] != (h, w):
        raise RuntimeError("Cooling image size mismatch vs heating images; ROI would be invalid.")

    cooling_feats_mean = compute_mean_features_for_sequence(cooling_files, mask)
    temps_cool, blocks_cool = group_by_temperature_features(cooling_feats_mean, TEMPS_COOLING, FRAMES_PER_TEMP)
    means_cool, stds_cool = compute_mean_std_features(blocks_cool)

    # ---- Frame-level true temps ----
    y_heat_frames = np.repeat(temps_heat, FRAMES_PER_TEMP).astype(float)
    y_cool_frames = np.repeat(temps_cool, FRAMES_PER_TEMP).astype(float)

    # ---- Mean-feature matrices ----
    X_heat_frames_mean, _ = select_feature_matrix(heating_feats_mean, USE_FEATURES)
    X_cool_frames_mean, _ = select_feature_matrix(cooling_feats_mean, USE_FEATURES)
    X_heat_means, _ = select_feature_matrix(means_heat, USE_FEATURES)
    X_cool_means, _ = select_feature_matrix(means_cool, USE_FEATURES)

    g_heat = y_heat_frames.astype(int)
    g_cool = y_cool_frames.astype(int)

    # ===========================
    # DEGREE SELECTION (stable + fast)
    # ===========================
    if AUTO_SELECT_DEGREE:
        deg_h = choose_degree_by_groupcv(X_heat_frames_mean, y_heat_frames, g_heat, POLY_DEGREE_CANDIDATES)
        deg_c = choose_degree_by_groupcv(X_cool_frames_mean, y_cool_frames, g_cool, POLY_DEGREE_CANDIDATES)
    else:
        deg_h = int(POLY_DEGREE)
        deg_c = int(POLY_DEGREE)

    # Global degree selection on mean features
    X_all_frames_mean = np.vstack([X_heat_frames_mean, X_cool_frames_mean])
    y_all_frames = np.concatenate([y_heat_frames, y_cool_frames])
    g_all = y_all_frames.astype(int)
    if AUTO_SELECT_DEGREE:
        deg_g = choose_degree_by_groupcv(X_all_frames_mean, y_all_frames, g_all, POLY_DEGREE_CANDIDATES)
    else:
        deg_g = int(POLY_DEGREE)

    # ===========================
    # BUILD TRAINING DATASETS (PIXEL SAMPLES)
    # ===========================
    if TRAIN_ON_PIXEL_SAMPLES:
        print("\nBuilding pixel-sample datasets (to match per-pixel inference)...")
        X_heat_train, y_heat_train = compute_pixel_samples_dataset(
            heating_files, mask, y_heat_frames,
            pixels_per_image=PIXELS_PER_IMAGE,
            max_total_samples=MAX_TOTAL_PIXEL_SAMPLES,
            seed=RANDOM_SEED,
            exclude_saturated=EXCLUDE_SATURATED_PIXELS,
            sat_thresh=SAT_THRESH_GRAY,
        )
        X_cool_train, y_cool_train = compute_pixel_samples_dataset(
            cooling_files, mask, y_cool_frames,
            pixels_per_image=PIXELS_PER_IMAGE,
            max_total_samples=MAX_TOTAL_PIXEL_SAMPLES,
            seed=RANDOM_SEED + 1,
            exclude_saturated=EXCLUDE_SATURATED_PIXELS,
            sat_thresh=SAT_THRESH_GRAY,
        )

        X_all_train = np.vstack([X_heat_train, X_cool_train])
        y_all_train = np.concatenate([y_heat_train, y_cool_train])

        print(f"  Heating pixel samples: {X_heat_train.shape[0]:,}")
        print(f"  Cooling pixel samples: {X_cool_train.shape[0]:,}")
        print(f"  Global  pixel samples: {X_all_train.shape[0]:,}")
    else:
        X_heat_train, y_heat_train = X_heat_frames_mean.astype(np.float32), y_heat_frames.astype(np.float32)
        X_cool_train, y_cool_train = X_cool_frames_mean.astype(np.float32), y_cool_frames.astype(np.float32)
        X_all_train, y_all_train = X_all_frames_mean.astype(np.float32), y_all_frames.astype(np.float32)

    # ===========================
    # FINAL MODEL: HEATING 
    # ===========================
    model_heat = make_huber_poly_model(deg_h)
    model_heat.fit(X_heat_train, y_heat_train)

    # Evaluate on per-image mean features 
    T_pred_heat_frames = model_heat.predict(X_heat_frames_mean)
    T_pred_heat_means = model_heat.predict(X_heat_means)

    report_metrics(y_heat_frames, T_pred_heat_frames, f"Heating FINAL model (frame-mean eval): deg={deg_h}")
    report_metrics(temps_heat, T_pred_heat_means, f"Heating FINAL model (per-temp means eval): deg={deg_h}")

    # ===========================
    # FINAL MODEL: COOLING 
    # ===========================
    model_cool = make_huber_poly_model(deg_c)
    model_cool.fit(X_cool_train, y_cool_train)

    T_pred_cool_frames = model_cool.predict(X_cool_frames_mean)
    T_pred_cool_means = model_cool.predict(X_cool_means)

    report_metrics(y_cool_frames, T_pred_cool_frames, f"Cooling FINAL model (frame-mean eval): deg={deg_c}")
    report_metrics(temps_cool, T_pred_cool_means, f"Cooling FINAL model (per-temp means eval): deg={deg_c}")

    # ===========================
    # FINAL MODEL: GLOBAL 
    # ===========================
    model_global = make_huber_poly_model(deg_g)
    model_global.fit(X_all_train, y_all_train)

    T_pred_all_frames = model_global.predict(X_all_frames_mean)

    uniq_T = np.array(sorted(set(g_all.tolist())), dtype=int)
    X_all_means = []
    y_all_means = []
    for t in uniq_T:
        sel = (g_all == t)
        X_all_means.append(np.mean(X_all_frames_mean[sel], axis=0))
        y_all_means.append(float(t))
    X_all_means = np.asarray(X_all_means, float)
    y_all_means = np.asarray(y_all_means, float)
    T_pred_all_means = model_global.predict(X_all_means)

    report_metrics(y_all_frames, T_pred_all_frames, f"GLOBAL FINAL model (frame-mean eval): deg={deg_g}")
    report_metrics(y_all_means, T_pred_all_means, f"GLOBAL FINAL model (per-temp means eval): deg={deg_g}")

    # ===========================
    # Save equations + metrics JSON 
    # ===========================
    eq_heat = polynomial_equation_string(model_heat, USE_FEATURES, precision=8)
    eq_cool = polynomial_equation_string(model_cool, USE_FEATURES, precision=8)
    eq_glob = polynomial_equation_string(model_global, USE_FEATURES, precision=8)

    eq_path = os.path.join(OUT_DIR, "equations_black_models_final.txt")
    with open(eq_path, "w", encoding="utf-8") as f:
        f.write("FINAL MODEL (HEATING)\n")
        f.write(f"Degree={deg_h}\n{eq_heat}\n\n")
        f.write("FINAL MODEL (COOLING)\n")
        f.write(f"Degree={deg_c}\n{eq_cool}\n\n")
        f.write("FINAL MODEL (GLOBAL / MERGED)\n")
        f.write(f"Degree={deg_g}\n{eq_glob}\n")

    huber_cfg = {
        "epsilon": float(HUBER_EPSILON),
        "alpha": float(HUBER_ALPHA),
        "max_iter": int(HUBER_MAX_ITER),
    }

    summary = {
        "output_dir": os.path.abspath(OUT_DIR),
        "use_features": list(USE_FEATURES),
        "frames_per_temp": int(FRAMES_PER_TEMP),
        "poly_degree_candidates": list(POLY_DEGREE_CANDIDATES),
        "auto_select_degree": bool(AUTO_SELECT_DEGREE),
        "huber": huber_cfg,
        "training_mode": "pixel_samples" if TRAIN_ON_PIXEL_SAMPLES else "frame_mean",
        "pixel_sampling": {
            "enabled": bool(TRAIN_ON_PIXEL_SAMPLES),
            "pixels_per_image": int(PIXELS_PER_IMAGE),
            "max_total_pixel_samples": int(MAX_TOTAL_PIXEL_SAMPLES),
            "exclude_saturated_pixels": bool(EXCLUDE_SATURATED_PIXELS),
            "sat_thresh_gray": int(SAT_THRESH_GRAY),
            "random_seed": int(RANDOM_SEED),
        },
        "models_final": {
            "heating": {
                "degree": int(deg_h),
                "equation": eq_heat,
                "metrics_frames": compute_metrics(y_heat_frames, T_pred_heat_frames),
                "metrics_means": compute_metrics(temps_heat, T_pred_heat_means),
            },
            "cooling": {
                "degree": int(deg_c),
                "equation": eq_cool,
                "metrics_frames": compute_metrics(y_cool_frames, T_pred_cool_frames),
                "metrics_means": compute_metrics(temps_cool, T_pred_cool_means),
            },
            "global": {
                "degree": int(deg_g),
                "equation": eq_glob,
                "metrics_frames": compute_metrics(y_all_frames, T_pred_all_frames),
                "metrics_means": compute_metrics(y_all_means, T_pred_all_means),
            },
        },
    }

    metrics_json_path = os.path.join(OUT_DIR, "models_final_summary_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ===========================
    # Grayscale arrays for plots 
    # ===========================
    g_heat_frames = heating_feats_mean[:, 3]
    g_cool_frames = cooling_feats_mean[:, 3]

    g_heat_means = means_heat[:, 3]
    g_cool_means = means_cool[:, 3]
    g_heat_stds = stds_heat[:, 3]
    g_cool_stds = stds_cool[:, 3]

    g_all_frames_feat = np.concatenate([g_heat_frames, g_cool_frames])

    # Global per-temp gray mean±std 
    g_all_means_feat = []
    g_all_stds_feat = []
    for t in uniq_T:
        sel = (g_all == t)
        g_all_means_feat.append(float(np.mean(g_all_frames_feat[sel])))
        g_all_stds_feat.append(float(np.std(g_all_frames_feat[sel], ddof=1)) if np.count_nonzero(sel) > 1 else 0.0)
    g_all_means_feat = np.asarray(g_all_means_feat, float)
    g_all_stds_feat = np.asarray(g_all_stds_feat, float)

    # ===========================
    # PLOTS 
    # ===========================
    plot_gray_vs_T_with_model(
        out_path=os.path.join(OUT_DIR, "01_heating_gray_vs_T_modelcurve.png"),
        title="Heating: Grayscale vs Temperature (measured points + model curve)",
        color_run=COLOR_HEAT,
        temps_true_frames=y_heat_frames,
        gray_frames=g_heat_frames,
        temps_true_means=temps_heat,
        gray_means=g_heat_means,
        gray_stds=g_heat_stds,
        T_pred_frames=T_pred_heat_frames,
        legend_model_label="Model trend (from T_pred)",
    )

    plot_gray_vs_T_with_model(
        out_path=os.path.join(OUT_DIR, "02_cooling_gray_vs_T_modelcurve.png"),
        title="Cooling: Grayscale vs Temperature (measured points + model curve)",
        color_run=COLOR_COOL,
        temps_true_frames=y_cool_frames,
        gray_frames=g_cool_frames,
        temps_true_means=temps_cool,
        gray_means=g_cool_means,
        gray_stds=g_cool_stds,
        T_pred_frames=T_pred_cool_frames,
        legend_model_label="Model trend (from T_pred)",
    )

    plot_hysteresis_gray_vs_T_models(
        out_path=os.path.join(OUT_DIR, "03_hysteresis_gray_vs_T_models.png"),
        temps_heat_means=temps_heat, g_heat_means=g_heat_means, g_heat_stds=g_heat_stds,
        T_pred_heat_frames=T_pred_heat_frames, g_heat_frames=g_heat_frames,
        temps_cool_means=temps_cool, g_cool_means=g_cool_means, g_cool_stds=g_cool_stds,
        T_pred_cool_frames=T_pred_cool_frames, g_cool_frames=g_cool_frames,
    )

    plot_05_global_true_vs_pred(
        out_path=os.path.join(OUT_DIR, "05_global_true_vs_pred.png"),
        y_heat_true=y_heat_frames,
        y_heat_pred=model_global.predict(X_heat_frames_mean),
        y_cool_true=y_cool_frames,
        y_cool_pred=model_global.predict(X_cool_frames_mean),
        y_means_true=y_all_means,
        y_means_pred=T_pred_all_means,
    )

    plot_global_gray_vs_T_all_modelcurve(
        out_path=os.path.join(OUT_DIR, "06_global_gray_vs_T_all_modelcurve.png"),
        y_heat_frames=y_heat_frames, g_heat_frames=g_heat_frames,
        y_cool_frames=y_cool_frames, g_cool_frames=g_cool_frames,
        uniq_T=uniq_T, g_all_means=g_all_means_feat, g_all_stds=g_all_stds_feat,
        T_pred_all_frames=T_pred_all_frames, g_all_frames=g_all_frames_feat,
    )

    # ===========================
    # Save models 
    # ===========================
    out_heat = out_cool = out_glob = None
    if joblib is not None:
        out_heat = os.path.join(OUT_DIR, f"black_model_heating_huber_deg{deg_h}.joblib")
        out_cool = os.path.join(OUT_DIR, f"black_model_cooling_huber_deg{deg_c}.joblib")
        out_glob = os.path.join(OUT_DIR, f"black_model_global_huber_deg{deg_g}.joblib")

        joblib.dump(
            {
                "model": model_heat,
                "use_features": USE_FEATURES,
                "poly_degree": deg_h,
                "regressor": "HuberRegressor",
                "scaler": "StandardScaler",
                "training_mode": "pixel_samples" if TRAIN_ON_PIXEL_SAMPLES else "frame_mean",
                "pixel_sampling": summary["pixel_sampling"],
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
                "training_mode": "pixel_samples" if TRAIN_ON_PIXEL_SAMPLES else "frame_mean",
                "pixel_sampling": summary["pixel_sampling"],
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
                "training_mode": "pixel_samples" if TRAIN_ON_PIXEL_SAMPLES else "frame_mean",
                "pixel_sampling": summary["pixel_sampling"],
            },
            out_glob
        )

    print("\nSaved:")
    print(f"  Output folder: {os.path.abspath(OUT_DIR)}")
    print(f"  ROI overlay: {overlay_path}")
    print(f"  Metrics JSON: {metrics_json_path}")
    print(f"  Equations: {eq_path}")
    if joblib is not None:
        print(f"  Models: {out_heat}, {out_cool}, {out_glob}")


if __name__ == "__main__":
    main()

