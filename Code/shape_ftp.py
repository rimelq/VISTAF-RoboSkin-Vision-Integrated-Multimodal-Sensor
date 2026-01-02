# shape_ftp.py

import os
import cv2
import heapq
import numpy as np
import json
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase
from mpl_toolkits.mplot3d import Axes3D 


# ===========================
# USER CONFIGURATION
# ===========================
REFERENCE_PATH = "./Final_demos_images/FINAL_reference.jpg"
DEFORMED_PATH  = "./Final_demos_images/FINAL_E_deformed.jpg"
OUTPUT_DIR     = "./Force/FTP/output"

CALIBRATION_JSON = "./Force/Phase_to_height/calibration_out/calibration_model.json"


# --- Sideband isolation method ---
FFT_SIDEBAND_METHOD = "patch_shift"  

# Patch-shift method parameters 
PATCH_HALF_WIDTH_BINS = 10
PATCH_WINDOW = "hann"             

BAND_RADIUS = 8
GAUSS_TRUNC_RADIUS = 24
DC_EXCLUSION = 10

# Padding before FFT
FFT_PAD_PX = 96

# Mild blur to suppress harmonics
PRE_BLUR_SIGMA_PX = 1.5

USE_FIXED_ROI = True
OUTER_CIRCLE_P1 = (1873, 1703)
OUTER_CIRCLE_P2 = (1599, 707)
OUTER_CIRCLE_P3 = (2575, 950)


# ===========================
# OUTPUT UNITS
# ===========================
OUTPUT_HEIGHT_IN_MM = True

# If True: indentations stay negative.
# If False: indentations become positive depth in mm.
MM_KEEP_INDENTATION_NEGATIVE = False


# ===========================
# CONTACT DEPTH THRESHOLD FILTER 
# ===========================
FILTER_SMALL_CONTACT_BLOBS = True

# remove a blob if its peak depth is below this (mm)
CONTACT_BLOB_MIN_PEAK_MM = 0.1
CONTACT_BLOB_MIN_PEAK_REL_FRAC = 1.0/3.0   

CONTACT_BLOB_MIN_AREA_PX = 0

CONTACT_BLOB_USE_CONTACT_D_MASK = False

# what to write where blobs are removed: "zero": set to baseline (0),
# "nan": hide completely (won't plot/export as values)
CONTACT_BLOB_REMOVED_VALUE = "zero"


# ===========================
# MIN-DEPTH DISPLAY 
# ===========================
SHOW_MIN_DEPTH_ON_HEIGHTMAP = False
MIN_DEPTH_SCOPE = "roi" 


# ===========================
# ROI / RELIABILITY CONFIG
# ===========================
SHOW_RELIABLE_ONLY_IN_HEIGHTMAP = False

ROI_ERODE_PX = 0
USE_CIRCULAR_APODIZATION = True
APOD_TAPER_PX = 120

AMP_VALID_PERCENTILE = 25.0
QUALITY_SMOOTH_SIGMA_PX = 6.0
RELIABLE_KEEP_LARGEST_CC = True
RELIABLE_EDGE_MARGIN_PX = 6

POLY_ORDER = 2
RELIABLE_SMOOTH_SIGMA_PX = 2.5


# ===========================
# FRONTIER (RELIABLE↔ROI) ZERO-TRANSITION
# ===========================
FRONTIER_ZERO_ENABLE = True
FRONTIER_ZERO_BAND_PX = 200           
FRONTIER_ZERO_CURVE = "smoothstep"   


# ===========================
# PREPROCESS and MASK FIXES
# ===========================
ILLUM_SIGMA_PX = 45
REMOVE_MEAN_AFTER_APOD = True

VALID_MORPH_CLOSE = True
VALID_CLOSE_KERNEL = 7
VALID_CLOSE_ITERS = 1

# Bad pixel glare removal before FFT
BAD_PIXEL_ENABLE = True
BAD_INTENSITY_PERCENTILE = 99.9
BAD_GRADIENT_PERCENTILE = 99.7
BAD_DILATE_KSIZE = 5
BAD_DILATE_ITERS = 1
BAD_INPAINT_RADIUS = 3
BAD_INPAINT_METHOD = "telea"       

# contact mask
CONTACT_CORE_PERCENTILE = 8.0
CONTACT_PERCENTILE = 92
DILATE_KERNEL_SIZE = 15
DILATE_ITERS = 2
MIN_CONTACT_FRAC = 0.002
MAX_CONTACT_FRAC = 0.40

USE_TWO_PASS_DETREND = True


UNRELIABLE_BASE_VALUE = 0.0

FILL_INTERNAL_HOLES_IN_RELIABLE = True
HOLE_NEIGHBORHOOD_PX = 11
HOLE_KNOWN_FRACTION = 0.70
HOLE_MIN_DIST_FROM_RELIABLE_EDGE_PX = 4

INPAINT_RADIUS = 5
INPAINT_METHOD = "telea"

SMOOTH_UNRELIABLE_REGION = True
UNRELIABLE_SMOOTH_SIGMA_PX = 9.0

# ===========================
# POSITIVE DEFORMATION CONTROL
# ===========================
ALLOW_POSITIVE_DEFORMATION = False


# --- height map matrix export ---
EXPORT_HEIGHTMAP_FILES = True
HEIGHTMAP_EXPORT_BASENAME = "height_map"
HEIGHTMAP_EXPORT_SAVE_CROP_CSV = True
HEIGHTMAP_EXPORT_SAVE_FULL_CSV = False


# ===========================
# DEBUG / BEHAVIOR CONFIG
# ===========================
DEBUG = True
DEBUG_LOG_TO_FILE = True
DEBUG_N_FFT_PEAKS = 12

BATCH_MODE = False              
SAVE_SUMMARY_FIGURES = True     

# --- Alignment ---
APPLY_GLOBAL_SHIFT = True

USE_ECC_CROP_ALIGNMENT = True
ECC_WARP_MODE = "euclidean"  
ECC_ITERS = 300
ECC_EPS = 1e-7
ECC_GAUSS_FILT = 5            


# grating-based pre-alignment
USE_GRATING_PREALIGNMENT = False

GRATING_PREALIGN_BAND_PX = 200

GRATING_PREALIGN_DILATE_RELIABLE_PX = 0

# High-pass filter strength for grating texture alignment
GRATING_PREALIGN_HP_SIGMA_PX = 35.0

# ECC settings for grating alignment
GRATING_PREALIGN_ECC_MODE = "euclidean" 
GRATING_PREALIGN_ECC_ITERS = 250
GRATING_PREALIGN_ECC_EPS = 1e-7
GRATING_PREALIGN_ECC_GAUSS_FILT = 0       


# --- Carrier handling ---
FORCE_RIGHT_HALF_PLANE = True
PREFER_PEAK_NEAR_CENTER_ROW = True
PEAK_MAX_DY_FROM_CENTER = 0.12

CARRIER_LOCAL_SEARCH_RADIUS = 6  

USE_HANN_WINDOW = False
AUTO_FLIP_SIGN = True

DEBUG_RAMP_DIAG = True
REMOVE_GLOBAL_PLANE_BEFORE_DETREND = True
PLANE_ORDER_FOR_REMOVAL = 1 

# ===========================
# CARRIER LOCK 
# ===========================
LOCK_CARRIER_TO_REFERENCE = True
APPLY_DK_RAMP_CORRECTION = True


# ===========================
# LOGGING
# ===========================
_LOG_FH = None


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def _open_log():
    global _LOG_FH
    if DEBUG and DEBUG_LOG_TO_FILE and _LOG_FH is None:
        _LOG_FH = open(os.path.join(OUTPUT_DIR, "debug_log.txt"), "w", encoding="utf-8")


def log(msg):
    if not DEBUG:
        return
    print(msg)
    if DEBUG_LOG_TO_FILE:
        _open_log()
        _LOG_FH.write(str(msg) + "\n")
        _LOG_FH.flush()


def close_log():
    global _LOG_FH
    if _LOG_FH is not None:
        _LOG_FH.close()
        _LOG_FH = None


def save_figure(fig, filename):
    full_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(full_path, dpi=300, bbox_inches="tight")
    log(f"Saved figure: {full_path}")


def export_heightmap_files(
    output_dir,
    basename,
    height_crop,
    height_full=None,
    crop_masks=None,
    full_masks=None,
    meta=None,
    save_crop_csv=True,
    save_full_csv=False
):
    ensure_output_dir(output_dir)

    crop_npy = os.path.join(output_dir, f"{basename}_crop.npy")
    np.save(crop_npy, height_crop.astype(np.float32))
    log(f"[EXPORT] Saved height map crop matrix: {crop_npy}")

    if height_full is not None:
        full_npy = os.path.join(output_dir, f"{basename}_full.npy")
        np.save(full_npy, height_full.astype(np.float32))
        log(f"[EXPORT] Saved height map full matrix: {full_npy}")

    if save_crop_csv:
        crop_csv = os.path.join(output_dir, f"{basename}_crop.csv")
        np.savetxt(crop_csv, height_crop.astype(np.float32), delimiter=",", fmt="%.9g")
        log(f"[EXPORT] Saved height map crop CSV: {crop_csv}")

    if save_full_csv and height_full is not None:
        full_csv = os.path.join(output_dir, f"{basename}_full.csv")
        np.savetxt(full_csv, height_full.astype(np.float32), delimiter=",", fmt="%.9g")
        log(f"[EXPORT] Saved height map full CSV: {full_csv}")

    bundle = {"height_crop": height_crop.astype(np.float32)}
    if height_full is not None:
        bundle["height_full"] = height_full.astype(np.float32)

    if crop_masks:
        for k, v in crop_masks.items():
            bundle[f"crop_{k}"] = np.asarray(v)

    if full_masks:
        for k, v in full_masks.items():
            bundle[f"full_{k}"] = np.asarray(v)

    if meta:
        for k, v in meta.items():
            bundle[f"meta_{k}"] = np.asarray(v)

    bundle_npz = os.path.join(output_dir, f"{basename}_bundle.npz")
    np.savez_compressed(bundle_npz, **bundle)
    log(f"[EXPORT] Saved height map bundle (npz): {bundle_npz}")


def array_stats(name, arr, mask=None):
    arr = np.asarray(arr)
    if mask is not None:
        vals = arr[mask]
        where = " (masked)"
    else:
        vals = arr.ravel()
        where = ""
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        log(f"[STATS] {name}{where}: EMPTY / no finite values")
        return
    q = np.quantile(vals, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    log(
        f"[STATS] {name}{where}: shape={arr.shape}, dtype={arr.dtype}, "
        f"min={q[0]:.6g}, p1={q[1]:.6g}, p5={q[2]:.6g}, "
        f"median={q[3]:.6g}, p95={q[4]:.6g}, p99={q[5]:.6g}, max={q[6]:.6g}, "
        f"mean={vals.mean():.6g}, std={vals.std():.6g}"
    )


def _finite_vals(arr, mask=None):
    if mask is None:
        v = np.asarray(arr).ravel()
    else:
        v = np.asarray(arr)[mask]
    v = v[np.isfinite(v)]
    return v


def _nanpercentile_safe(arr, q, mask=None, fallback=None):
    v = _finite_vals(arr, mask)
    if v.size == 0:
        return fallback
    return float(np.nanpercentile(v, q))


def _nanmedian_safe(arr, mask=None, fallback=None):
    v = _finite_vals(arr, mask)
    if v.size == 0:
        return fallback
    return float(np.nanmedian(v))


# ===========================
# ROI HELPERS
# ===========================
def select_circular_roi(image_bgr):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.set_title("Select circular ROI: click CENTER, then a point on the EDGE.\nClose the window after clicks.")
    plt.axis("off")

    log("Click the CENTER of the circular ROI, then a point on its EDGE.")
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)

    if len(pts) < 2:
        raise RuntimeError("ROI selection aborted (fewer than 2 clicks).")

    (x1, y1), (x2, y2) = pts
    cx = int(round(x1))
    cy = int(round(y1))
    radius = int(round(np.hypot(x2 - x1, y2 - y1)))

    log(f"Selected circular ROI center=({cx}, {cy}), radius={radius} pixels.")
    return cx, cy, radius


def create_circular_mask(h, w, center_x, center_y, radius):
    Y, X = np.ogrid[:h, :w]
    dist_sq = (X - center_x) ** 2 + (Y - center_y) ** 2
    return dist_sq <= radius ** 2


def create_circular_apodization(h, w, cx, cy, r, taper_px):
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    apo = np.zeros((h, w), np.float32)
    r_in = max(0.0, float(r - taper_px))

    inside_flat = d <= r_in
    inside_taper = (d > r_in) & (d <= r)

    apo[inside_flat] = 1.0
    if taper_px > 0:
        t = (d[inside_taper] - r_in) / max(1e-6, float(taper_px))
        apo[inside_taper] = 0.5 * (1.0 + np.cos(np.pi * t))
    return apo


def circle_from_3_points(p1, p2, p3):
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    A = np.array([[2*(x2-x1), 2*(y2-y1)],
                  [2*(x3-x1), 2*(y3-y1)]], dtype=float)
    b = np.array([x2*x2 + y2*y2 - x1*x1 - y1*y1,
                  x3*x3 + y3*y3 - x1*x1 - y1*y1], dtype=float)
    cx, cy = np.linalg.solve(A, b)
    r = float(np.hypot(cx - x1, cy - y1))
    return int(round(cx)), int(round(cy)), int(round(r))


# ===========================
# FFT PEAKS
# ===========================
def find_top_peaks(mag, dc_exclusion, n_peaks=10):
    mag = np.asarray(mag)
    h, w = mag.shape
    cy, cx = h // 2, w // 2

    mag_for_search = mag.copy()
    y0 = max(0, cy - dc_exclusion)
    y1 = min(h, cy + dc_exclusion)
    x0 = max(0, cx - dc_exclusion)
    x1 = min(w, cx + dc_exclusion)
    mag_for_search[y0:y1, x0:x1] = 0

    flat = mag_for_search.ravel()
    n_peaks = min(n_peaks, flat.size)
    idx = np.argpartition(flat, -n_peaks)[-n_peaks:]
    idx = idx[np.argsort(flat[idx])[::-1]]

    peaks = []
    for i in idx:
        y, x = np.unravel_index(i, mag_for_search.shape)
        peaks.append((int(x), int(y), float(mag_for_search[y, x])))
    return peaks


def choose_carrier_peak(peaks, h, w):
    cy, cx = h // 2, w // 2
    candidates = peaks[:]

    if FORCE_RIGHT_HALF_PLANE:
        c2 = [p for p in candidates if p[0] > cx]
        if len(c2) > 0:
            candidates = c2

    if PREFER_PEAK_NEAR_CENTER_ROW:
        max_dy = int(PEAK_MAX_DY_FROM_CENTER * h)
        c2 = [p for p in candidates if abs(p[1] - cy) <= max_dy]
        if len(c2) > 0:
            candidates = c2

    if len(candidates) == 0:
        candidates = peaks

    carrier = max(candidates, key=lambda t: t[2])
    return carrier[0], carrier[1]


def _parabolic_subpixel_1d(fm1, f0, fp1):
    den = (fm1 - 2.0 * f0 + fp1)
    if abs(den) < 1e-12:
        return 0.0
    return 0.5 * (fm1 - fp1) / den


def refine_peak_parabolic_log(mag, peak_x, peak_y):
    h, w = mag.shape
    x = int(peak_x)
    y = int(peak_y)
    if x <= 0 or x >= w - 1 or y <= 0 or y >= h - 1:
        return float(x), float(y)

    lm = np.log(mag + 1e-12)
    dx = _parabolic_subpixel_1d(lm[y, x - 1], lm[y, x], lm[y, x + 1])
    dy = _parabolic_subpixel_1d(lm[y - 1, x], lm[y, x], lm[y + 1, x])
    return float(x + dx), float(y + dy)


def refine_peak_local_max(fft_mag, x_guess, y_guess, radius=6):
    h, w = fft_mag.shape
    xg = int(np.clip(int(round(x_guess)), 0, w - 1))
    yg = int(np.clip(int(round(y_guess)), 0, h - 1))
    r = int(max(1, radius))

    x0 = max(1, xg - r)
    x1 = min(w - 2, xg + r)
    y0 = max(1, yg - r)
    y1 = min(h - 2, yg + r)

    win = fft_mag[y0:y1+1, x0:x1+1]
    iy, ix = np.unravel_index(np.argmax(win), win.shape)
    xp = x0 + ix
    yp = y0 + iy

    xf, yf = refine_peak_parabolic_log(fft_mag, xp, yp)
    return (xp, yp), (xf, yf)


def plot_fft_with_peaks(fft_mag, peaks, chosen_peak, title):
    h, w = fft_mag.shape
    cy, cx = h // 2, w // 2

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(np.log1p(fft_mag), cmap="gray")
    ax.scatter([cx], [cy], s=30, marker="x")
    ax.set_title(title)
    ax.axis("off")

    for i, (px, py, pm) in enumerate(peaks[:min(len(peaks), 12)]):
        ax.scatter([px], [py], s=18)
        ax.text(px + 5, py + 5, f"{i}", fontsize=8)

    px, py = chosen_peak
    ax.scatter([px], [py], s=80, facecolors="none", edgecolors="r", linewidths=2)
    ax.text(px + 8, py + 8, "chosen", color="r", fontsize=9)
    return fig


# ===========================
# ALIGNMENT
# ===========================
def estimate_global_shift(ref_gray_full, def_gray_full):
    ref_blur = cv2.GaussianBlur(ref_gray_full, (0, 0), 7)
    def_blur = cv2.GaussianBlur(def_gray_full, (0, 0), 7)
    h, w = ref_blur.shape
    hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(np.float32(ref_blur), np.float32(def_blur), hann)
    return shift, response


def _ecc_mode_from_string(s):
    s = str(s).lower().strip()
    if s == "translation":
        return cv2.MOTION_TRANSLATION
    if s == "euclidean":
        return cv2.MOTION_EUCLIDEAN
    if s == "affine":
        return cv2.MOTION_AFFINE
    raise ValueError(f"Unknown ECC_WARP_MODE: {s}")


def align_crop_ecc(ref_u8, mov_u8, mask_bool=None, mode="euclidean", iters=300, eps=1e-7, gauss_filt=5):
    warp_mode = _ecc_mode_from_string(mode)
    h, w = ref_u8.shape

    ref = ref_u8.astype(np.float32) / 255.0
    mov = mov_u8.astype(np.float32) / 255.0

    if gauss_filt and gauss_filt > 0:
        ref = cv2.GaussianBlur(ref, (0, 0), gauss_filt)
        mov = cv2.GaussianBlur(mov, (0, 0), gauss_filt)

    warp = np.eye(2, 3, dtype=np.float32)

    m = None
    if mask_bool is not None:
        m = (mask_bool.astype(np.uint8) * 255)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(iters), float(eps))

    try:
        cc, warp = cv2.findTransformECC(ref, mov, warp, warp_mode, criteria, inputMask=m, gaussFiltSize=1)
        aligned = cv2.warpAffine(
            mov_u8, warp, (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT
        )
        return aligned, warp, float(cc)
    except cv2.error as e:
        log(f"[ECC] ERROR: {e}")
        return mov_u8, warp, float("nan")


def warp_affine_any(img, warp):
    h, w = img.shape[:2]
    return cv2.warpAffine(
        img, warp, (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT
    )


def highpass_to_u8(img_u8, sigma_px, mask=None):
    img = img_u8.astype(np.float32)
    low = cv2.GaussianBlur(img, (0, 0), float(max(0.0, sigma_px))) if sigma_px and sigma_px > 0 else 0.0
    hp = img - low

    if mask is not None and np.any(mask):
        v = hp[mask]
    else:
        v = hp.ravel()
    v = v[np.isfinite(v)]
    if v.size < 50:
        # fallback: just normalize whole hp
        v = hp.ravel()

    p1 = np.percentile(v, 1.0)
    p99 = np.percentile(v, 99.0)
    if not np.isfinite(p1) or not np.isfinite(p99) or (p99 - p1) < 1e-6:
        out = np.zeros_like(img_u8, dtype=np.uint8)
        return out

    x = (hp - p1) / (p99 - p1)
    x = np.clip(x, 0.0, 1.0)
    return (255.0 * x).astype(np.uint8)


# ===========================
# BAD PIXEL / GLARE PREPROCESS
# ===========================
def _safe_percentile(vals, q, fallback):
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return fallback
    return float(np.percentile(vals, q))


def detect_bad_pixels(gray_f32, valid_mask=None):
    img = gray_f32.astype(np.float32)
    if valid_mask is None:
        valid_mask = np.isfinite(img)

    v = img[valid_mask]
    hi_thr = _safe_percentile(v, BAD_INTENSITY_PERCENTILE, fallback=np.max(v) if v.size else 255.0)

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx*gx + gy*gy)
    g_thr = _safe_percentile(grad[valid_mask], BAD_GRADIENT_PERCENTILE, fallback=np.max(grad) if v.size else 0.0)

    bad = (img >= hi_thr) | (grad >= g_thr)
    bad &= valid_mask

    if BAD_DILATE_KSIZE and BAD_DILATE_KSIZE > 1:
        ksz = int(BAD_DILATE_KSIZE)
        ksz = max(3, ksz | 1)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        bad_u8 = (bad.astype(np.uint8) * 255)
        bad_u8 = cv2.dilate(bad_u8, k, iterations=int(BAD_DILATE_ITERS))
        bad = bad_u8 > 0

    return bad


def inpaint_float32(img_f32, mask_bool, radius=3, method="telea"):
    if not np.any(mask_bool):
        return img_f32

    img = img_f32.astype(np.float32).copy()
    if np.any(~np.isfinite(img)):
        med = float(np.nanmedian(img[np.isfinite(img)])) if np.any(np.isfinite(img)) else 0.0
        img[~np.isfinite(img)] = med

    m = np.zeros(img.shape, np.uint8)
    m[mask_bool] = 255

    flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    out = cv2.inpaint(img, m, float(radius), flag).astype(np.float32)
    return out


# ===========================
# CONNECTED COMPONENT UTILS
# ===========================
def load_calibration(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        cal = json.load(f)

    # Your phase_to_height.py writes:
    # cal["best_model"] = {"type":..., "params":..., ...}
    model = cal["best_model"]
    use_neg = bool(cal.get("use_negated_height_for_fit", True))
    return model, use_neg

def model_predict(model, xs):
    xs = np.asarray(xs, float)
    t = model["type"]
    p = model["params"]

    xs = np.maximum(xs, 0.0)

    if t == "growth":
        a = float(p["a"]); b = float(p["b"])
        return a * (np.exp(b * xs) - 1.0)

    if t == "hinge_saturating":
        a = float(p["a"]); b = float(p["b"]); c = float(p["c"])
        return a * (
            (1.0 - np.exp(-b * np.maximum(xs - c, 0.0)))
            - (1.0 - np.exp(-b * np.maximum(0.0 - c, 0.0)))
        )

    raise ValueError(f"Unknown model type in calibration: {t}")

def height_unitless_to_depth_mm(height_unitless, model, use_negated_height=True):
    h = np.asarray(height_unitless, dtype=np.float32)
    x = (-h) if use_negated_height else h
    return model_predict(model, x).astype(np.float32)

def largest_connected_component(mask_bool):
    m = mask_bool.astype(np.uint8)
    if np.count_nonzero(m) == 0:
        return mask_bool

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return mask_bool

    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    return labels == best


def erode_by_distance(mask_bool, margin_px):
    if margin_px <= 0:
        return mask_bool
    m = (mask_bool.astype(np.uint8) * 255)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    return (dist > float(margin_px)) & mask_bool


def dilate_mask(mask_bool, px):
    if px is None or px <= 0:
        return mask_bool
    ksz = int(max(3, (2 * int(px) + 1)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    u8 = (mask_bool.astype(np.uint8) * 255)
    u8 = cv2.dilate(u8, k, iterations=1)
    return (u8 > 0)


def compute_reliable_mask(amp_ref, amp_def, roi_eroded, circ_mask, tag=""):
    prefix = f"[{tag}] " if tag else ""

    amp_prod = (amp_ref * amp_def).astype(np.float32)
    quality = amp_prod

    if QUALITY_SMOOTH_SIGMA_PX and QUALITY_SMOOTH_SIGMA_PX > 0:
        quality = cv2.GaussianBlur(quality, (0, 0), QUALITY_SMOOTH_SIGMA_PX).astype(np.float32)
        log(f"{prefix}[QUAL] Smoothed amp_prod sigma={QUALITY_SMOOTH_SIGMA_PX:g}px")

    amp_thr = _nanpercentile_safe(quality, AMP_VALID_PERCENTILE, mask=roi_eroded, fallback=None)
    if amp_thr is None:
        amp_thr = _nanpercentile_safe(quality, AMP_VALID_PERCENTILE, mask=circ_mask, fallback=0.0)

    reliable = roi_eroded & (quality >= float(amp_thr)) & np.isfinite(quality)

    if VALID_MORPH_CLOSE and np.any(reliable):
        ksz = int(VALID_CLOSE_KERNEL)
        ksz = max(3, ksz | 1)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        rel_u8 = (reliable.astype(np.uint8) * 255)
        rel_u8 = cv2.morphologyEx(rel_u8, cv2.MORPH_CLOSE, k, iterations=int(VALID_CLOSE_ITERS))
        reliable = (rel_u8 > 0) & roi_eroded
        log(f"{prefix}[MASK] Applied morph-close to reliable (ksz={ksz}, iters={VALID_CLOSE_ITERS})")

    if RELIABLE_KEEP_LARGEST_CC and np.any(reliable):
        reliable = largest_connected_component(reliable) & roi_eroded
        log(f"{prefix}[MASK] Kept largest connected component of reliable mask.")

    if RELIABLE_EDGE_MARGIN_PX and RELIABLE_EDGE_MARGIN_PX > 0 and np.any(reliable):
        reliable = erode_by_distance(reliable, RELIABLE_EDGE_MARGIN_PX)
        log(f"{prefix}[MASK] Distance-eroded reliable by {RELIABLE_EDGE_MARGIN_PX}px.")

    rel_pct = 100.0 * (reliable.sum() / max(1, roi_eroded.sum()))
    log(f"{prefix}[MASK] amp_thr(p{AMP_VALID_PERCENTILE:.1f})={amp_thr:.6g}, reliable-within-ROI%={rel_pct:.3f}%")

    return reliable, quality, float(amp_thr), amp_prod


def compute_between_roi_and_reliable_mask(roi_eroded, reliable, band_px=0, dilate_reliable_px=0):
    roi = roi_eroded.astype(bool)
    rel = (reliable.astype(bool) & roi)

    if dilate_reliable_px and dilate_reliable_px > 0:
        rel = dilate_mask(rel, int(dilate_reliable_px)) & roi

    outside = roi & (~rel)

    if band_px and band_px > 0 and np.any(rel):
        dt_input = np.full(rel.shape, 255, np.uint8)
        dt_input[rel] = 0
        dist = cv2.distanceTransform(dt_input, cv2.DIST_L2, 3).astype(np.float32)
        dist_edge = np.maximum(dist - 1.0, 0.0)
        outside = outside & (dist_edge <= float(band_px))

    return outside


# ===========================
# FTP CORE
# ===========================
def _make_patch_window(hp, wp, kind="hann"):
    if kind == "none":
        return np.ones((hp, wp), np.float32)
    if kind.lower() == "hann":
        wy = np.hanning(hp).astype(np.float32)
        wx = np.hanning(wp).astype(np.float32)
        return (wy[:, None] * wx[None, :]).astype(np.float32)
    return np.ones((hp, wp), np.float32)


def ftp_complex_demod(
    gray_crop_u8,
    band_radius,
    dc_exclusion,
    carrier_peak=None,
    carrier_peak_refined=None,
    carrier_local_search_radius=0,
    apo_mask=None,
    tag="",
    lock_carrier_to_ref=False
):
    img0 = gray_crop_u8.astype(np.float32)
    array_stats(f"{tag} gray_crop", img0)

    if BAD_PIXEL_ENABLE:
        valid = (apo_mask > 1e-6) if apo_mask is not None else np.ones_like(img0, dtype=bool)
        bad = detect_bad_pixels(img0, valid_mask=valid)
        if np.any(bad):
            img0 = inpaint_float32(img0, bad, radius=BAD_INPAINT_RADIUS, method=BAD_INPAINT_METHOD)
            log(f"[BAD] {tag} inpainted bad pixels: {int(bad.sum())} px")

    blur = cv2.GaussianBlur(img0, (0, 0), sigmaX=ILLUM_SIGMA_PX, sigmaY=ILLUM_SIGMA_PX)
    I_norm = img0 / (blur + 1e-6) - 1.0
    array_stats(f"{tag} I_norm (pre-apo)", I_norm)

    if PRE_BLUR_SIGMA_PX and PRE_BLUR_SIGMA_PX > 0:
        I_norm = cv2.GaussianBlur(I_norm, (0, 0), PRE_BLUR_SIGMA_PX).astype(np.float32)
        log(f"[PREBLUR] {tag} Applied pre-blur sigma={PRE_BLUR_SIGMA_PX:g}px")

    Iw = I_norm
    if apo_mask is not None:
        Iw = Iw * apo_mask
        log(f"[APOD] {tag} Applied circular apodization.")

    if REMOVE_MEAN_AFTER_APOD:
        m = (apo_mask > 1e-6) if apo_mask is not None else None
        mu = _nanmedian_safe(Iw, mask=m, fallback=0.0)
        Iw = Iw - mu
        log(f"[DC] {tag} Removed mean after apodization: {mu:.6g}")

    if USE_HANN_WINDOW:
        h0, w0 = Iw.shape
        wy = np.hanning(h0).astype(np.float32)
        wx = np.hanning(w0).astype(np.float32)
        Iw = Iw * (wy[:, None] * wx[None, :])
        log(f"[WIN] {tag} Applied Hann window.")

    pad = int(max(0, FFT_PAD_PX))
    if pad > 0:
        Iw_fft = cv2.copyMakeBorder(Iw, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT)
        log(f"[PAD] {tag} FFT padding: {pad}px -> {Iw_fft.shape[0]}x{Iw_fft.shape[1]}")
    else:
        Iw_fft = Iw

    hf, wf = Iw_fft.shape
    cy, cx = hf // 2, wf // 2

    F = np.fft.fft2(Iw_fft)
    F_shift = np.fft.fftshift(F)
    fft_mag = np.abs(F_shift)
    array_stats(f"{tag} fft_mag", fft_mag)

    peaks = find_top_peaks(fft_mag, dc_exclusion=dc_exclusion, n_peaks=DEBUG_N_FFT_PEAKS)
    if DEBUG:
        log(f"[FFT] {tag} Top peaks (x, y, mag):")
        for i, (px, py, pm) in enumerate(peaks):
            log(f"  {i:02d}: ({px:4d}, {py:4d}), mag={pm:.6g}, dx={px-cx}, dy={py-cy}")

    if carrier_peak is None:
        peak_x, peak_y = choose_carrier_peak(peaks, h=hf, w=wf)
        log(f"[FFT] {tag} Chosen carrier peak (int) = (x={peak_x}, y={peak_y}), dx={peak_x-cx}, dy={peak_y-cy}")

        if carrier_peak_refined is None:
            peak_x_f, peak_y_f = refine_peak_parabolic_log(fft_mag, peak_x, peak_y)
        else:
            peak_x_f, peak_y_f = float(carrier_peak_refined[0]), float(carrier_peak_refined[1])

    else:
        gx, gy = float(carrier_peak[0]), float(carrier_peak[1])
        log(f"[FFT] {tag} Carrier GUESS = (x={gx:.3f}, y={gy:.3f})")

        if lock_carrier_to_ref and (carrier_peak_refined is not None):
            peak_x_f, peak_y_f = float(carrier_peak_refined[0]), float(carrier_peak_refined[1])
            peak_x, peak_y = int(np.round(peak_x_f)), int(np.round(peak_y_f))
            log(f"[FFT] {tag} LOCKED carrier to REF refined = ({peak_x_f:.3f},{peak_y_f:.3f})")
        else:
            if carrier_local_search_radius and carrier_local_search_radius > 0:
                (peak_x, peak_y), (peak_x_f, peak_y_f) = refine_peak_local_max(
                    fft_mag, gx, gy, radius=int(carrier_local_search_radius)
                )
                log(f"[FFT] {tag} Local-refined peak: int=({peak_x},{peak_y}) refined=({peak_x_f:.3f},{peak_y_f:.3f}) "
                    f"dx={peak_x-cx}, dy={peak_y-cy}")
            else:
                peak_x, peak_y = int(round(gx)), int(round(gy))
                peak_x_f, peak_y_f = refine_peak_parabolic_log(fft_mag, peak_x, peak_y)
                log(f"[FFT] {tag} Using PROVIDED carrier peak int=({peak_x},{peak_y}) refined=({peak_x_f:.3f},{peak_y_f:.3f})")

    kx = peak_x_f - cx
    ky = peak_y_f - cy
    log(f"[FFT] {tag} Carrier k (bins): kx={kx:.3f}, ky={ky:.3f}")

    if abs(kx) > 1e-9:
        est_period_px = wf / abs(kx)
        log(f"[FFT] {tag} Estimated grating period ≈ {est_period_px:.3f} px")

    if DEBUG:
        fig = plot_fft_with_peaks(
            fft_mag, peaks,
            (int(round(peak_x_f)), int(round(peak_y_f))),
            title=f"{tag} FFT (log) with detected peaks"
        )
        save_figure(fig, f"DEBUG_fft_peaks_{tag}.png")
        plt.close(fig)

    # ---------------------------
    # SIDEBAND ISOLATION + DEMOD
    # ---------------------------
    fft_mask = np.zeros((hf, wf), np.float32)

    if FFT_SIDEBAND_METHOD.lower() == "patch_shift":
        px_i = int(np.round(peak_x_f))
        py_i = int(np.round(peak_y_f))

        bw = int(max(3, PATCH_HALF_WIDTH_BINS))
        x0 = max(0, px_i - bw)
        x1 = min(wf, px_i + bw + 1)
        y0 = max(0, py_i - bw)
        y1 = min(hf, py_i + bw + 1)

        patch = F_shift[y0:y1, x0:x1].copy()
        ph, pw = patch.shape

        win = _make_patch_window(ph, pw, kind=PATCH_WINDOW)
        patch *= win

        F_demod_shift = np.zeros_like(F_shift)
        cy0 = cy - ph // 2
        cx0 = cx - pw // 2
        F_demod_shift[cy0:cy0+ph, cx0:cx0+pw] = patch

        fft_mask[y0:y1, x0:x1] = win

        F_demod = np.fft.ifftshift(F_demod_shift)
        complex_field = np.fft.ifft2(F_demod)

        dpx = float(peak_x_f - px_i)
        dpy = float(peak_y_f - py_i)
        if abs(dpx) > 1e-6 or abs(dpy) > 1e-6:
            yy, xx = np.mgrid[0:hf, 0:wf]
            frac_demod = np.exp(-1j * 2.0 * np.pi * (dpx * (xx / wf) + dpy * (yy / hf)))
            complex_field = complex_field * frac_demod
            log(f"[PATCH] {tag} fractional correction: dpx={dpx:.4f}, dpy={dpy:.4f}")

        log(f"[PATCH] {tag} patch {ph}x{pw} (bw={bw}) window={PATCH_WINDOW}")
        complex_demod_full = complex_field

    else:
        Y, X = np.ogrid[:hf, :wf]
        dist2_peak = (X - peak_x_f) ** 2 + (Y - peak_y_f) ** 2
        dist2_dc = (X - cx) ** 2 + (Y - cy) ** 2

        sigma = float(max(1e-6, band_radius))
        gauss = np.exp(-0.5 * dist2_peak / (sigma * sigma)).astype(np.float32)

        rcut = float(max(3.0, GAUSS_TRUNC_RADIUS))
        gauss *= (dist2_peak <= (rcut * rcut)).astype(np.float32)

        gauss[dist2_dc <= float(dc_exclusion * dc_exclusion)] = 0.0
        fft_mask = gauss

        F_filt_shift = F_shift * fft_mask
        F_filt = np.fft.ifftshift(F_filt_shift)
        complex_field = np.fft.ifft2(F_filt)

        yy, xx = np.mgrid[0:hf, 0:wf]
        demod = np.exp(-1j * 2.0 * np.pi * (kx * (xx / wf) + ky * (yy / hf)))
        complex_demod_full = complex_field * demod

        log(f"[GAUSS] {tag} sigma={sigma:.3g}, truncR={rcut:.3g}, dc_notch={dc_exclusion}")

    if pad > 0:
        complex_demod = complex_demod_full[pad:pad+Iw.shape[0], pad:pad+Iw.shape[1]]
        I_fft_input = Iw_fft[pad:pad+Iw.shape[0], pad:pad+Iw.shape[1]]
    else:
        complex_demod = complex_demod_full
        I_fft_input = Iw_fft

    amp2 = np.abs(complex_demod).astype(np.float32)

    if DEBUG:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(amp2, cmap="gray")
        ax.set_title(f"{tag} Complex amplitude |ifft| (demod)")
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.7)
        save_figure(fig, f"DEBUG_complex_amplitude_{tag}.png")
        plt.close(fig)

        phw = np.angle(complex_demod).astype(np.float32)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(phw, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax.set_title(f"{tag} Wrapped phase (demod)")
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.7)
        save_figure(fig, f"DEBUG_phase_wrapped_{tag}.png")
        plt.close(fig)

        phu = unwrap_phase(phw)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(phu, cmap="viridis")
        ax.set_title(f"{tag} Unwrapped phase (demod) [debug only]")
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.7)
        save_figure(fig, f"DEBUG_phase_unwrapped_{tag}.png")
        plt.close(fig)

    return (
        I_norm,
        I_fft_input,
        fft_mag,
        fft_mask,
        complex_demod,
        (int(round(peak_x_f)), int(round(peak_y_f))),
        (float(peak_x_f), float(peak_y_f)),
        amp2,
        (float(kx), float(ky)),
        (hf, wf)
    )


# ===========================
# MASKED UNWRAP
# ===========================
def unwrap_quality_guided(wrapped, mask, quality):
    h, w = wrapped.shape
    unwrapped = np.full((h, w), np.nan, np.float32)
    m = mask.astype(bool)
    if not np.any(m):
        return unwrapped

    q = quality.copy().astype(np.float32)
    q[~m] = -np.inf
    sy, sx = np.unravel_index(np.argmax(q), q.shape)
    unwrapped[sy, sx] = wrapped[sy, sx]

    heap = []
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def push_neighbors(py, px):
        for dy, dx in nbrs:
            ny, nx = py + dy, px + dx
            if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and not np.isfinite(unwrapped[ny, nx]):
                heapq.heappush(heap, (-float(q[ny, nx]), ny, nx, py, px))

    push_neighbors(sy, sx)

    while heap:
        _, y, x, py, px = heapq.heappop(heap)
        if not m[y, x]:
            continue
        if np.isfinite(unwrapped[y, x]):
            continue
        if not np.isfinite(unwrapped[py, px]):
            continue

        dw = np.angle(np.exp(1j * (wrapped[y, x] - wrapped[py, px])))
        unwrapped[y, x] = unwrapped[py, px] + dw
        push_neighbors(y, x)

    return unwrapped


# ===========================
# ROBUST 2D POLY FIT 
# ===========================
def build_design_matrix(xn, yn, order):
    cols = [xn, yn, np.ones_like(xn)]
    if order >= 2:
        cols += [xn * xn, xn * yn, yn * yn]
    return np.stack(cols, axis=1)


def eval_poly2d(xn, yn, coef, order):
    z = coef[0] * xn + coef[1] * yn + coef[2]
    if order >= 2:
        z = z + coef[3] * xn * xn + coef[4] * xn * yn + coef[5] * yn * yn
    return z


def robust_polyfit2d(z, mask, order=2, iters=6, c=4.685):
    h, w = z.shape
    m = mask & np.isfinite(z)
    if np.count_nonzero(m) < 200:
        ncoef = 6 if order >= 2 else 3
        return np.zeros((ncoef,), np.float32), np.zeros_like(z, np.float32)

    yy, xx = np.indices((h, w))
    x = xx[m].astype(np.float32)
    y = yy[m].astype(np.float32)
    zz = z[m].astype(np.float32)

    xn = (x - (w - 1) / 2.0) / ((w - 1) / 2.0)
    yn = (y - (h - 1) / 2.0) / ((h - 1) / 2.0)

    A = build_design_matrix(xn, yn, order)
    wts = np.ones_like(zz, np.float32)

    for _ in range(iters):
        Aw = A * wts[:, None]
        zw = zz * wts
        coef, *_ = np.linalg.lstsq(Aw, zw, rcond=None)

        r = zz - (A @ coef)
        med = np.median(r)
        mad = np.median(np.abs(r - med)) + 1e-6
        sigma = 1.4826 * mad
        u = r / (c * sigma)
        wts = 1.0 / (1.0 + u * u)

    coef = coef.astype(np.float32)

    yyf, xxf = np.indices((h, w))
    xnf = (xxf.astype(np.float32) - (w - 1) / 2.0) / ((w - 1) / 2.0)
    ynf = (yyf.astype(np.float32) - (h - 1) / 2.0) / ((h - 1) / 2.0)
    fit = eval_poly2d(xnf, ynf, coef, order).astype(np.float32)
    return coef, fit


def masked_gaussian_smooth(z, mask, sigma=1.2):
    if sigma <= 0:
        return z
    z0 = z.copy().astype(np.float32)
    m = mask.astype(np.float32)
    z0[~mask] = 0.0
    num = cv2.GaussianBlur(z0, (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma) + 1e-6
    return (num / den).astype(np.float32)


# ===========================
# HOLE CANDIDATES INSIDE RELIABLE
# ===========================
def compute_internal_holes_within_mask(container_mask, known_mask, ksize, frac_thr, min_dist_edge_px):
    container = container_mask.astype(bool)
    known = known_mask.astype(bool) & container
    holes = container & (~known)
    if not np.any(holes):
        return np.zeros_like(container, dtype=bool)

    k = int(ksize)
    k = max(3, k | 1)

    cont_f = container.astype(np.float32)
    known_f = known.astype(np.float32)

    count_known = cv2.boxFilter(known_f, ddepth=-1, ksize=(k, k), normalize=False)
    count_cont = cv2.boxFilter(cont_f, ddepth=-1, ksize=(k, k), normalize=False)

    frac = count_known / (count_cont + 1e-6)

    cont_u8 = (container.astype(np.uint8) * 255)
    dist = cv2.distanceTransform(cont_u8, cv2.DIST_L2, 3)

    candidates = holes & (frac >= float(frac_thr)) & (dist >= float(min_dist_edge_px))
    return candidates


def inpaint_only_mask(z_known, roi_mask, inpaint_mask, radius=5, method="telea"):
    z = z_known.astype(np.float32)
    roi = roi_mask.astype(bool)
    m = inpaint_mask.astype(bool) & roi

    out = z.copy()
    out[~roi] = np.nan

    if not np.any(m):
        return out

    known = roi & (~m) & np.isfinite(z)
    fill_val = float(np.nanmedian(z[known])) if np.any(known) else 0.0

    zin = np.full_like(z, fill_val, dtype=np.float32)
    zin[known] = z[known]

    mask_u8 = np.zeros(z.shape, np.uint8)
    mask_u8[m] = 255

    flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    zout = cv2.inpaint(zin, mask_u8, float(radius), flag).astype(np.float32)

    out[m] = zout[m]
    out[~roi] = np.nan
    return out


def clamp_positive_to_zero(z, mask=None):
    out = z.astype(np.float32).copy()
    if mask is None:
        m = np.isfinite(out)
    else:
        m = mask.astype(bool) & np.isfinite(out)
    out[m] = np.minimum(out[m], 0.0)
    return out

def filter_blobs_by_peak_depth_mm(
    height_mm,
    roi_mask,
    min_peak_mm=0.1,
    min_area_px=0,
    removed_value="zero",
    restrict_mask=None,
    min_peak_rel_frac=None,   # <-- NEW
):
    out = height_mm.astype(np.float32).copy()
    roi = roi_mask.astype(bool) & np.isfinite(out)

    if restrict_mask is not None:
        roi &= restrict_mask.astype(bool)

    depth = (-out if MM_KEEP_INDENTATION_NEGATIVE else out).astype(np.float32)

    cand = roi & (depth > 0.0)
    if not np.any(cand):
        return out, np.zeros_like(roi_mask, dtype=bool)

    global_max_peak = float(np.max(depth[cand]))

    thr = float(min_peak_mm)
    if (min_peak_rel_frac is not None) and np.isfinite(global_max_peak):
        thr = max(thr, float(min_peak_rel_frac) * global_max_peak)

    log(f"[CONTACT-FILTER] global_max_peak={global_max_peak:.6g} mm, threshold={thr:.6g} mm")

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        cand.astype(np.uint8), connectivity=8
    )

    keep_labels = np.zeros((num,), dtype=bool)

    kept_mask = np.zeros_like(cand, dtype=bool)
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if min_area_px and area < int(min_area_px):
            continue

        comp = (labels == lab)
        peak = float(np.max(depth[comp])) if np.any(comp) else 0.0

        if peak >= thr:
            keep_labels[lab] = True
            kept_mask |= comp

    removed = cand & (~kept_mask)

    if removed_value.lower() == "nan":
        out[removed] = np.nan
    else:
        out[removed] = 0.0 

    log(f"[CONTACT-FILTER] blobs total={num-1}, kept={int(keep_labels.sum())}, removed_px={int(removed.sum())}")
    return out, kept_mask


# ===========================
# FRONTIER ZERO-TRANSITION HELPERS
# ===========================
def _curve01(t, kind="smoothstep"):
    t = np.clip(t, 0.0, 1.0).astype(np.float32)
    k = str(kind).lower().strip()
    if k == "linear":
        return t
    if k == "cosine":
        return (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def apply_frontier_zero_transition(
    height,
    roi_mask,
    reliable_mask,
    band_px,
    curve="smoothstep",
    base_value=0.0,
    apply_inside=True,
    apply_outside=True
):
    out = height.astype(np.float32).copy()
    roi = roi_mask.astype(bool)
    rel = reliable_mask.astype(bool) & roi

    if (not np.any(rel)) or (band_px is None) or (float(band_px) <= 0):
        return out

    band = float(band_px)

    rel_u8 = (rel.astype(np.uint8) * 255)
    inv_u8 = ((~rel).astype(np.uint8) * 255)

    dist_in = cv2.distanceTransform(rel_u8, cv2.DIST_L2, 3).astype(np.float32)
    dist_in_edge = np.maximum(dist_in - 1.0, 0.0)

    dist_out = cv2.distanceTransform(inv_u8, cv2.DIST_L2, 3).astype(np.float32)
    dist_out_edge = np.maximum(dist_out - 1.0, 0.0)

    if apply_inside:
        inside = rel & np.isfinite(out)
        w = _curve01(dist_in_edge / max(1e-6, band), kind=curve)
        out[inside] = float(base_value) + (out[inside] - float(base_value)) * w[inside]

    if apply_outside:
        outside_band = roi & (~rel) & (dist_out_edge <= band)
        out[outside_band] = float(base_value)

    return out


# ===========================
# 3D PLOT
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
    ax.set_zlabel("height (mm)" if OUTPUT_HEIGHT_IN_MM else "height (arb. units)")
    fig.colorbar(surf, shrink=0.6, label=("mm" if OUTPUT_HEIGHT_IN_MM else "height"))
    return fig


# ===========================
# DEBUG: ramp diagnostics
# ===========================
def debug_ramp(phase_unwrapped, reliable, tag="phase"):
    if not DEBUG_RAMP_DIAG:
        return phase_unwrapped

    h, w = phase_unwrapped.shape
    yy, xx = np.indices((h, w))

    vals = phase_unwrapped[reliable]
    if vals.size < 500:
        log("[RAMP] Not enough points for ramp debug.")
        return phase_unwrapped

    vx = xx[reliable].astype(np.float32)
    vy = yy[reliable].astype(np.float32)
    v = vals.astype(np.float32)
    v = v - np.nanmedian(v)
    vx = vx - np.nanmedian(vx)
    vy = vy - np.nanmedian(vy)
    cxr = float(np.dot(v, vx) / (np.linalg.norm(v) * np.linalg.norm(vx) + 1e-9))
    cyr = float(np.dot(v, vy) / (np.linalg.norm(v) * np.linalg.norm(vy) + 1e-9))
    log(f"[RAMP] corr(phase, x)={cxr:.3f}, corr(phase, y)={cyr:.3f} (|corr| near 1 => strong ramp)")

    coef, fit = robust_polyfit2d(phase_unwrapped, reliable, order=int(PLANE_ORDER_FOR_REMOVAL))
    res = (phase_unwrapped - fit).astype(np.float32)

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    a0 = ax[0].imshow(phase_unwrapped, cmap="coolwarm")
    ax[0].set_title(f"{tag}: unwrapped (masked)")
    ax[0].axis("off")
    fig.colorbar(a0, ax=ax[0], shrink=0.7)

    a1 = ax[1].imshow(fit, cmap="viridis")
    ax[1].set_title(f"{tag}: fitted plane/order{PLANE_ORDER_FOR_REMOVAL}")
    ax[1].axis("off")
    fig.colorbar(a1, ax=ax[1], shrink=0.7)

    a2 = ax[2].imshow(res, cmap="coolwarm")
    ax[2].set_title(f"{tag}: residual after fit")
    ax[2].axis("off")
    fig.colorbar(a2, ax=ax[2], shrink=0.7)

    fig.tight_layout()
    save_figure(fig, f"DEBUG_ramp_{tag}.png")
    plt.close(fig)

    cy0 = h // 2
    row = phase_unwrapped[cy0, :].copy()
    row_fit = fit[cy0, :].copy()
    row_res = res[cy0, :].copy()

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(row, label="phase")
    ax.plot(row_fit, label="fit")
    ax.plot(row_res, label="residual")
    ax.set_title(f"{tag}: center-row cross-section")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, f"DEBUG_ramp_cross_{tag}.png")
    plt.close(fig)

    if REMOVE_GLOBAL_PLANE_BEFORE_DETREND:
        log(f"[RAMP] Removing fitted order{PLANE_ORDER_FOR_REMOVAL} component before detrend.")
        out = phase_unwrapped - fit
        return out.astype(np.float32)

    return phase_unwrapped


# ===========================
# MAIN
# ===========================
def main(
    reference_path=None,
    deformed_path=None,
    output_dir=None,
    calibration_json=None,
    batch_mode=False,
    save_summary_figures=True,
    export_heightmaps=None,
    debug=None,
    return_results=False
):
    
    global REFERENCE_PATH, DEFORMED_PATH, OUTPUT_DIR, CALIBRATION_JSON
    global BATCH_MODE, SAVE_SUMMARY_FIGURES, EXPORT_HEIGHTMAP_FILES
    global DEBUG, DEBUG_LOG_TO_FILE

    if reference_path is not None:
        REFERENCE_PATH = reference_path
    if deformed_path is not None:
        DEFORMED_PATH = deformed_path
    if output_dir is not None:
        OUTPUT_DIR = output_dir
    if calibration_json is not None:
        CALIBRATION_JSON = calibration_json

    BATCH_MODE = bool(batch_mode)
    SAVE_SUMMARY_FIGURES = bool(save_summary_figures)

    if export_heightmaps is not None:
        EXPORT_HEIGHTMAP_FILES = bool(export_heightmaps)

    if debug is not None:
        DEBUG = bool(debug)
        if not DEBUG:
            DEBUG_LOG_TO_FILE = False


    ensure_output_dir(OUTPUT_DIR)

    cal_model, cal_use_neg = load_calibration(CALIBRATION_JSON)
    
    log("=== FTP DEBUG RUN START ===")

    ref_bgr = cv2.imread(REFERENCE_PATH, cv2.IMREAD_COLOR)
    def_bgr = cv2.imread(DEFORMED_PATH, cv2.IMREAD_COLOR)
    if ref_bgr is None:
        raise RuntimeError(f"Could not read reference image: {REFERENCE_PATH}")
    if def_bgr is None:
        raise RuntimeError(f"Could not read deformed image: {DEFORMED_PATH}")
    if ref_bgr.shape != def_bgr.shape:
        raise RuntimeError("Reference and deformed images have different sizes.")

    H, W = ref_bgr.shape[:2]
    log(f"[IMG] Ref shape={ref_bgr.shape}, Def shape={def_bgr.shape}, HxW={H}x{W}")

    ref_gray_full = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    def_gray_full = cv2.cvtColor(def_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    shift, response = estimate_global_shift(ref_gray_full, def_gray_full)
    log(f"[ALIGN] phaseCorrelate estimated shift (dx, dy)=({shift[0]:.4f}, {shift[1]:.4f}), response={response:.6g}")

    if APPLY_GLOBAL_SHIFT:
        M = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], dtype=np.float32)
        def_bgr = cv2.warpAffine(def_bgr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        log("[ALIGN] Applied global shift correction to deformed image.")

    log("Images loaded. Select CIRCULAR ROI on reference image.")

    if USE_FIXED_ROI:
        cx_full, cy_full, r_full = circle_from_3_points(OUTER_CIRCLE_P1, OUTER_CIRCLE_P2, OUTER_CIRCLE_P3)
        log(f"[ROI] Fixed ROI from 3 points: center=({cx_full},{cy_full}), r={r_full}")
    else:
        cx_full, cy_full, r_full = select_circular_roi(ref_bgr)

    x1 = max(0, cx_full - r_full)
    x2 = min(W, cx_full + r_full)
    y1 = max(0, cy_full - r_full)
    y2 = min(H, cy_full + r_full)
    log(f"[ROI] Bounding box: x=[{x1},{x2}), y=[{y1},{y2}) -> crop={y2-y1}x{x2-x1}")

    ref_crop = ref_bgr[y1:y2, x1:x2]
    def_crop = def_bgr[y1:y2, x1:x2]

    ref_gray = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)
    def_gray = cv2.cvtColor(def_crop, cv2.COLOR_BGR2GRAY)

    h, w = ref_gray.shape
    cx_local = cx_full - x1
    cy_local = cy_full - y1
    r_local = int(min(r_full, cx_local, cy_local, w - 1 - cx_local, h - 1 - cy_local))

    circ_mask = create_circular_mask(h, w, cx_local, cy_local, r_local)
    log(f"[ROI] Local center=({cx_local},{cy_local}), r={r_local}, mask%={100*circ_mask.mean():.3f}%")

    r_valid = max(0, r_local - int(ROI_ERODE_PX))
    roi_eroded = create_circular_mask(h, w, cx_local, cy_local, r_valid)
    log(f"[ROI] Eroded ROI radius={r_valid} (erode={ROI_ERODE_PX}px), roi-eroded%={100*roi_eroded.mean():.3f}%")

    apo = None
    if USE_CIRCULAR_APODIZATION:
        apo = create_circular_apodization(h, w, cx_local, cy_local, r_local, APOD_TAPER_PX)

    if USE_ECC_CROP_ALIGNMENT:
        aligned_def, warp, cc = align_crop_ecc(
            ref_gray, def_gray, mask_bool=circ_mask,
            mode=ECC_WARP_MODE, iters=ECC_ITERS, eps=ECC_EPS, gauss_filt=ECC_GAUSS_FILT
        )
        def_gray = aligned_def
        def_crop = warp_affine_any(def_crop, warp)
        log(f"[ECC] mode={ECC_WARP_MODE}, cc={cc:.6g}, warp=\n{warp}")

    # ===========================
    # Optional grating-based prealignment on band between ROI and reliable
    # ===========================
    if USE_GRATING_PREALIGNMENT:
        log("[GRATING-ALIGN] Pass-1 FTP to estimate reliable mask for alignment region...")

        # Reference demod
        Iref_norm_1, Iref_fft_in_1, fft_ref_mag_1, mask_ref_1, cref_1, peak_ref_int_1, peak_ref_f_1, amp_ref_1, k_ref_1, hf_wf_ref_1 = ftp_complex_demod(
            ref_gray,
            band_radius=BAND_RADIUS,
            dc_exclusion=DC_EXCLUSION,
            apo_mask=apo,
            tag="ref_prealign",
            lock_carrier_to_ref=False
        )

        # Deformed demod
        if LOCK_CARRIER_TO_REFERENCE:
            Idef_norm_1, Idef_fft_in_1, fft_def_mag_1, mask_def_1, cdef_1, peak_def_int_1, peak_def_f_1, amp_def_1, k_def_1, hf_wf_def_1 = ftp_complex_demod(
                def_gray,
                band_radius=BAND_RADIUS,
                dc_exclusion=DC_EXCLUSION,
                carrier_peak=peak_ref_int_1,
                carrier_peak_refined=peak_ref_f_1,
                carrier_local_search_radius=0,
                apo_mask=apo,
                tag="def_prealign",
                lock_carrier_to_ref=True
            )
        else:
            Idef_norm_1, Idef_fft_in_1, fft_def_mag_1, mask_def_1, cdef_1, peak_def_int_1, peak_def_f_1, amp_def_1, k_def_1, hf_wf_def_1 = ftp_complex_demod(
                def_gray,
                band_radius=BAND_RADIUS,
                dc_exclusion=DC_EXCLUSION,
                carrier_peak=peak_ref_int_1,
                carrier_peak_refined=None,
                carrier_local_search_radius=CARRIER_LOCAL_SEARCH_RADIUS,
                apo_mask=apo,
                tag="def_prealign",
                lock_carrier_to_ref=False
            )

        reliable_1, quality_1, amp_thr_1, _ = compute_reliable_mask(amp_ref_1, amp_def_1, roi_eroded, circ_mask, tag="PREALIGN")

        align_mask = compute_between_roi_and_reliable_mask(
            roi_eroded=roi_eroded,
            reliable=reliable_1,
            band_px=int(GRATING_PREALIGN_BAND_PX),
            dilate_reliable_px=int(GRATING_PREALIGN_DILATE_RELIABLE_PX)
        )

        if np.any(align_mask):
            ref_hp = highpass_to_u8(ref_gray, GRATING_PREALIGN_HP_SIGMA_PX, mask=align_mask)
            def_hp = highpass_to_u8(def_gray, GRATING_PREALIGN_HP_SIGMA_PX, mask=align_mask)

            if DEBUG:
                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                ax[0].imshow(align_mask, cmap="gray"); ax[0].set_title("Grating prealign mask"); ax[0].axis("off")
                ax[1].imshow(ref_hp, cmap="gray"); ax[1].set_title("Ref HP (for grating align)"); ax[1].axis("off")
                ax[2].imshow(def_hp, cmap="gray"); ax[2].set_title("Def HP (for grating align)"); ax[2].axis("off")
                fig.tight_layout()
                save_figure(fig, "DEBUG_grating_prealign_inputs.png")
                plt.close(fig)

            _, warp_g, cc_g = align_crop_ecc(
                ref_hp, def_hp, mask_bool=align_mask,
                mode=GRATING_PREALIGN_ECC_MODE,
                iters=GRATING_PREALIGN_ECC_ITERS,
                eps=GRATING_PREALIGN_ECC_EPS,
                gauss_filt=GRATING_PREALIGN_ECC_GAUSS_FILT
            )

            def_gray = warp_affine_any(def_gray, warp_g)
            def_crop = warp_affine_any(def_crop, warp_g)

            log(f"[GRATING-ALIGN] Applied grating ECC warp: mode={GRATING_PREALIGN_ECC_MODE}, cc={cc_g:.6g}, warp=\n{warp_g}")
        else:
            log("[GRATING-ALIGN] align_mask is empty; skipping grating prealignment.")

    if SAVE_SUMMARY_FIGURES:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(ref_gray, cmap="gray"); ax[0].contour(circ_mask, colors="r", linewidths=0.5)
        ax[0].set_title("Reference crop + ROI"); ax[0].axis("off")
        ax[1].imshow(def_gray, cmap="gray"); ax[1].contour(circ_mask, colors="r", linewidths=0.5)
        ax[1].set_title("Deformed crop (aligned) + ROI"); ax[1].axis("off")
        fig.tight_layout()
        save_figure(fig, "03_ref_def_crops_with_roi.png")
        plt.close(fig)

    # ===========================
    # FULL RUN
    # ===========================
    log("Running FTP on reference crop...")
    Iref_norm, Iref_fft_in, fft_ref_mag, mask_ref, cref, peak_ref_int, peak_ref_f, amp_ref, k_ref, hf_wf_ref = ftp_complex_demod(
        ref_gray,
        band_radius=BAND_RADIUS,
        dc_exclusion=DC_EXCLUSION,
        apo_mask=apo,
        tag="ref",
        lock_carrier_to_ref=False
    )

    if LOCK_CARRIER_TO_REFERENCE:
        log("Running FTP on deformed crop (CARRIER LOCKED to reference refined peak)...")
        Idef_norm, Idef_fft_in, fft_def_mag, mask_def, cdef, peak_def_int, peak_def_f, amp_def, k_def, hf_wf_def = ftp_complex_demod(
            def_gray,
            band_radius=BAND_RADIUS,
            dc_exclusion=DC_EXCLUSION,
            carrier_peak=peak_ref_int,
            carrier_peak_refined=peak_ref_f,
            carrier_local_search_radius=0,
            apo_mask=apo,
            tag="def",
            lock_carrier_to_ref=True
        )
    else:
        log("Running FTP on deformed crop (carrier refined locally around reference peak)...")
        Idef_norm, Idef_fft_in, fft_def_mag, mask_def, cdef, peak_def_int, peak_def_f, amp_def, k_def, hf_wf_def = ftp_complex_demod(
            def_gray,
            band_radius=BAND_RADIUS,
            dc_exclusion=DC_EXCLUSION,
            carrier_peak=peak_ref_int,
            carrier_peak_refined=None,
            carrier_local_search_radius=CARRIER_LOCAL_SEARCH_RADIUS,
            apo_mask=apo,
            tag="def",
            lock_carrier_to_ref=False
        )

    dkx = k_def[0] - k_ref[0]
    dky = k_def[1] - k_ref[1]
    log(f"[CARRIER] k_ref={k_ref}, k_def={k_def}, Δk=(dkx={dkx:.3f}, dky={dky:.3f}) bins")
    hf, wf = hf_wf_def
    ramp_x = 2.0 * np.pi * dkx * (w - 1) / max(1, wf)
    ramp_y = 2.0 * np.pi * dky * (h - 1) / max(1, hf)
    log(f"[CARRIER] Expected residual ramp across crop: Δphi_x≈{ramp_x:.3f} rad, Δphi_y≈{ramp_y:.3f} rad")

    reliable, quality, amp_thr, amp_prod = compute_reliable_mask(amp_ref, amp_def, roi_eroded, circ_mask, tag="FINAL")
    if not np.any(reliable):
        log("[ERROR] reliable mask is empty. Lower AMP_VALID_PERCENTILE or RELIABLE_EDGE_MARGIN_PX or ROI_ERODE_PX.")
        return

    ratio = cdef * np.conj(cref)

    if APPLY_DK_RAMP_CORRECTION and ((abs(dkx) > 1e-6) or (abs(dky) > 1e-6)):
        yy, xx = np.mgrid[0:h, 0:w]
        ramp = np.exp(1j * 2.0 * np.pi * (dkx * (xx / max(1, wf)) + dky * (yy / max(1, hf))))
        ratio = ratio * ramp
        log(f"[CARRIER] Applied dk-ramp correction on ratio (dkx={dkx:.6g}, dky={dky:.6g}).")

    phase_diff_wrapped = np.angle(ratio).astype(np.float32)

    if DEBUG:
        tmp = phase_diff_wrapped.copy()
        tmp[~reliable] = np.nan
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(tmp, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax.set_title("DEBUG phase_diff_wrapped (reliable)")
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.7)
        save_figure(fig, "DEBUG_phase_diff_wrapped.png")
        plt.close(fig)

    phase_diff_unwrapped = unwrap_quality_guided(phase_diff_wrapped, reliable, quality=quality)

    phase_vis_dbg = phase_diff_unwrapped.copy()
    phase_vis_dbg[~reliable] = np.nan
    phase_diff_unwrapped = debug_ramp(phase_vis_dbg, reliable, tag="phase_diff")

    if not USE_TWO_PASS_DETREND:
        coef, fit = robust_polyfit2d(phase_diff_unwrapped, reliable, order=POLY_ORDER)
        phase_detrended = (phase_diff_unwrapped - fit).astype(np.float32)
        bg_med = _nanmedian_safe(phase_detrended, mask=reliable, fallback=0.0)
        phase_zeroed = phase_detrended - float(bg_med)
        contact_d = np.zeros_like(reliable, dtype=bool)
        log("[DETREND] Single-pass robust detrend over reliable.")
    else:
        coef0, fit0 = robust_polyfit2d(phase_diff_unwrapped, reliable, order=POLY_ORDER)
        residual0 = (phase_diff_unwrapped - fit0).astype(np.float32)

        abs_res = np.abs(residual0).astype(np.float32)
        thr = _nanpercentile_safe(abs_res, CONTACT_PERCENTILE, mask=reliable, fallback=None)
        if thr is None or not np.isfinite(thr):
            thr = _nanpercentile_safe(abs_res, 95, mask=reliable, fallback=0.0)

        contact = (abs_res >= float(thr)) & reliable & np.isfinite(abs_res)

        frac = contact.sum() / max(1, reliable.sum())
        if frac < MIN_CONTACT_FRAC:
            thr2 = _nanpercentile_safe(abs_res, 95, mask=reliable, fallback=thr)
            contact = (abs_res >= float(thr2)) & reliable & np.isfinite(abs_res)
        elif frac > MAX_CONTACT_FRAC:
            thr2 = _nanpercentile_safe(abs_res, 98, mask=reliable, fallback=thr)
            contact = (abs_res >= float(thr2)) & reliable & np.isfinite(abs_res)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE))
        contact_u8 = (contact.astype(np.uint8) * 255)
        contact_d = (cv2.dilate(contact_u8, k, iterations=DILATE_ITERS) > 0) & reliable

        background = reliable & (~contact_d)
        if background.sum() < int(0.15 * reliable.sum()):
            log("[MASK] background too small; using reliable as background for detrend.")
            background = reliable.copy()

        coef, fit = robust_polyfit2d(phase_diff_unwrapped, background, order=POLY_ORDER)
        phase_detrended = (phase_diff_unwrapped - fit).astype(np.float32)

        bg_med = _nanmedian_safe(phase_detrended, mask=background, fallback=None)
        if bg_med is None or not np.isfinite(bg_med):
            bg_med = _nanmedian_safe(phase_detrended, mask=reliable, fallback=0.0)

        phase_zeroed = phase_detrended - float(bg_med)
        log("[DETREND] Two-pass detrend with contact/background separation.")

    height_map = phase_zeroed.copy()

    if RELIABLE_SMOOTH_SIGMA_PX and RELIABLE_SMOOTH_SIGMA_PX > 0:
        height_map = masked_gaussian_smooth(height_map, reliable & np.isfinite(height_map), sigma=RELIABLE_SMOOTH_SIGMA_PX)
        log(f"[SMOOTH] reliable-only smoothing sigma={RELIABLE_SMOOTH_SIGMA_PX:g}px")

    if AUTO_FLIP_SIGN and np.any(reliable):
        core_thr = _nanpercentile_safe(height_map, CONTACT_CORE_PERCENTILE, mask=reliable, fallback=None)
        if core_thr is not None and np.isfinite(core_thr):
            core = reliable & np.isfinite(height_map) & (height_map <= float(core_thr))
            if np.any(core):
                med_core = float(np.median(height_map[core]))
                log(f"[SIGN] median(height) in contact_core = {med_core:.6g}")
                if med_core > 0:
                    height_map *= -1.0
                    log("[SIGN] Flipped sign so indentation becomes negative.")

    known_height = reliable & np.isfinite(height_map)

    hole_candidates = np.zeros((h, w), dtype=bool)
    height_rel_filled = np.full((h, w), np.nan, np.float32)
    height_rel_filled[known_height] = height_map[known_height]

    if FILL_INTERNAL_HOLES_IN_RELIABLE:
        hole_candidates = compute_internal_holes_within_mask(
            container_mask=reliable,
            known_mask=known_height,
            ksize=HOLE_NEIGHBORHOOD_PX,
            frac_thr=HOLE_KNOWN_FRACTION,
            min_dist_edge_px=HOLE_MIN_DIST_FROM_RELIABLE_EDGE_PX
        )
        log(f"[HOLES] candidates inside reliable = {hole_candidates.sum()} px")

        if np.any(hole_candidates):
            tmp = height_rel_filled.copy()
            med = float(np.nanmedian(tmp[known_height])) if np.any(known_height) else 0.0
            tmp[reliable & ~known_height] = med

            filled = inpaint_only_mask(
                z_known=tmp,
                roi_mask=reliable,
                inpaint_mask=hole_candidates,
                radius=INPAINT_RADIUS,
                method=INPAINT_METHOD
            )
            height_rel_filled[hole_candidates] = filled[hole_candidates]
            log(f"[HOLES] inpainted holes (method={INPAINT_METHOD}, radius={INPAINT_RADIUS})")

    output_reliable = reliable & np.isfinite(height_rel_filled)

    if FRONTIER_ZERO_ENABLE and FRONTIER_ZERO_BAND_PX and FRONTIER_ZERO_BAND_PX > 0:
        height_rel_filled = apply_frontier_zero_transition(
            height_rel_filled,
            roi_mask=roi_eroded,
            reliable_mask=output_reliable,
            band_px=FRONTIER_ZERO_BAND_PX,
            curve=FRONTIER_ZERO_CURVE,
            base_value=UNRELIABLE_BASE_VALUE,
            apply_inside=True,
            apply_outside=False
        )
        log(f"[FRONTIER] Applied inside-reliable taper to 0 (band={FRONTIER_ZERO_BAND_PX}px, curve={FRONTIER_ZERO_CURVE}).")

    height_final = np.full((h, w), np.nan, np.float32)
    height_final[roi_eroded] = float(UNRELIABLE_BASE_VALUE)
    height_final[output_reliable] = height_rel_filled[output_reliable]

    if SMOOTH_UNRELIABLE_REGION and UNRELIABLE_SMOOTH_SIGMA_PX and UNRELIABLE_SMOOTH_SIGMA_PX > 0:
        smooth_all = masked_gaussian_smooth(height_final, roi_eroded, sigma=UNRELIABLE_SMOOTH_SIGMA_PX)
        upd = roi_eroded & (~output_reliable)
        height_final[upd] = smooth_all[upd]
        log(f"[SMOOTH] unreliable-region smoothing sigma={UNRELIABLE_SMOOTH_SIGMA_PX:g}px (reliable preserved)")

    if FRONTIER_ZERO_ENABLE and FRONTIER_ZERO_BAND_PX and FRONTIER_ZERO_BAND_PX > 0:
        height_final = apply_frontier_zero_transition(
            height_final,
            roi_mask=roi_eroded,
            reliable_mask=output_reliable,
            band_px=FRONTIER_ZERO_BAND_PX,
            curve=FRONTIER_ZERO_CURVE,
            base_value=UNRELIABLE_BASE_VALUE,
            apply_inside=False,
            apply_outside=True
        )
        log(f"[FRONTIER] Enforced outside band to base=0 (band={FRONTIER_ZERO_BAND_PX}px).")

    if not ALLOW_POSITIVE_DEFORMATION:
        height_final = clamp_positive_to_zero(height_final, mask=roi_eroded)
        log("[CLAMP] ALLOW_POSITIVE_DEFORMATION=False -> clamped positive heights to 0 in ROI.")

    # ===========================
    # CONVERT UNITLESS HEIGHT -> MM (using calibration)
    # ===========================
    height_final_unitless = height_final

    height_final_out = height_final_unitless

    if OUTPUT_HEIGHT_IN_MM:
        depth_mm = height_unitless_to_depth_mm(height_final_unitless, cal_model, cal_use_neg)  # >= 0
        if MM_KEEP_INDENTATION_NEGATIVE:
            height_final_out = -depth_mm  
        else:
            height_final_out = depth_mm    

    # ===========================
    # Remove shallow blobs by PEAK depth (mm)
    # ===========================
    contact_kept = np.zeros_like(roi_eroded, dtype=bool)

    if FILTER_SMALL_CONTACT_BLOBS and OUTPUT_HEIGHT_IN_MM:
        restrict = contact_d if CONTACT_BLOB_USE_CONTACT_D_MASK else None

        height_final_out, contact_kept = filter_blobs_by_peak_depth_mm(
            height_mm=height_final_out,
            roi_mask=roi_eroded,
            min_peak_mm=CONTACT_BLOB_MIN_PEAK_MM,
            min_peak_rel_frac=CONTACT_BLOB_MIN_PEAK_REL_FRAC,  
            min_area_px=CONTACT_BLOB_MIN_AREA_PX,
            removed_value=CONTACT_BLOB_REMOVED_VALUE,
            restrict_mask=restrict
        )

    if EXPORT_HEIGHTMAP_FILES:
        height_full = np.full((H, W), np.nan, np.float32)
        height_full[y1:y2, x1:x2] = height_final_out

        roi_full = np.zeros((H, W), dtype=bool)
        reliable_full = np.zeros((H, W), dtype=bool)
        output_reliable_full = np.zeros((H, W), dtype=bool)
        circ_full = np.zeros((H, W), dtype=bool)
        hole_full = np.zeros((H, W), dtype=bool)
        contact_full = np.zeros((H, W), dtype=bool)

        roi_full[y1:y2, x1:x2] = roi_eroded
        reliable_full[y1:y2, x1:x2] = reliable
        output_reliable_full[y1:y2, x1:x2] = output_reliable
        circ_full[y1:y2, x1:x2] = circ_mask
        contact_kept_full = np.zeros((H, W), dtype=bool)
        contact_kept_full[y1:y2, x1:x2] = contact_kept
        hole_full[y1:y2, x1:x2] = hole_candidates
        contact_full[y1:y2, x1:x2] = contact_d

        export_heightmap_files(
            output_dir=OUTPUT_DIR,
            basename=HEIGHTMAP_EXPORT_BASENAME,
            height_crop=height_final_out,
            height_full=height_full,
            crop_masks={
                "roi_eroded": roi_eroded,
                "reliable": reliable,
                "output_reliable": output_reliable,
                "circ_mask": circ_mask,
                "contact_kept_by_depth": contact_kept,
                "hole_candidates": hole_candidates,
                "contact_dilated": contact_d,
            },
            full_masks={
                "roi_eroded": roi_full,
                "reliable": reliable_full,
                "output_reliable": output_reliable_full,
                "circ_mask": circ_full,
                "contact_kept_by_depth": contact_kept_full,
                "hole_candidates": hole_full,
                "contact_dilated": contact_full,
            },
            meta={
                "crop_x1": np.int32(x1),
                "crop_y1": np.int32(y1),
                "crop_x2": np.int32(x2),
                "crop_y2": np.int32(y2),
                "roi_center_x_full": np.int32(cx_full),
                "roi_center_y_full": np.int32(cy_full),
                "roi_radius_full": np.int32(r_full),
                "roi_center_x_crop": np.int32(cx_local),
                "roi_center_y_crop": np.int32(cy_local),
                "roi_radius_crop": np.int32(r_local),
            },
            save_crop_csv=HEIGHTMAP_EXPORT_SAVE_CROP_CSV,
            save_full_csv=HEIGHTMAP_EXPORT_SAVE_FULL_CSV
        )

    phase_vis = phase_diff_unwrapped.copy()
    phase_vis[~output_reliable] = np.nan

    height_vis = height_final_out.copy()
    height_vis[~roi_eroded] = np.nan
    if SHOW_RELIABLE_ONLY_IN_HEIGHTMAP:
        height_vis[~output_reliable] = np.nan

    min_val = np.nan
    min_xy = None

    if SHOW_MIN_DEPTH_ON_HEIGHTMAP:
        if str(MIN_DEPTH_SCOPE).lower().strip() == "reliable":
            mmin = output_reliable & np.isfinite(height_final_out)
        else:
            mmin = roi_eroded & np.isfinite(height_final_out)

        if np.any(mmin):
            vals = height_final_out.copy()
            vals[~mmin] = np.nan
            if MM_KEEP_INDENTATION_NEGATIVE:
                iy, ix = np.unravel_index(np.nanargmin(vals), vals.shape)
            else:
                iy, ix = np.unravel_index(np.nanargmax(vals), vals.shape)
            min_val = float(height_final_out[iy, ix])
            min_xy = (int(ix), int(iy))

    if SAVE_SUMMARY_FIGURES:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im0 = axes[0].imshow(phase_vis, cmap="coolwarm")
        axes[0].contour(circ_mask, colors="k", linewidths=0.5)
        axes[0].set_title("Phase diff (unwrapped, OUTPUT-RELIABLE only)")
        axes[0].axis("off")
        fig.colorbar(im0, ax=axes[0], shrink=0.7)

        im1 = axes[1].imshow(height_vis, cmap="viridis")

        if SHOW_MIN_DEPTH_ON_HEIGHTMAP and min_xy is not None and np.isfinite(min_val):
            axes[1].plot([min_xy[0]], [min_xy[1]], marker="x", markersize=10, markeredgewidth=2)
            axes[1].text(
                0.02, 0.98,
                f"min = {min_val:.6g} mm\n(x={min_xy[0]}, y={min_xy[1]})",
                transform=axes[1].transAxes,
                ha="left", va="top",
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none")
            )

        axes[1].contour(circ_mask, colors="k", linewidths=0.5)
        if SHOW_MIN_DEPTH_ON_HEIGHTMAP and np.isfinite(min_val):
            axes[1].set_title(f"Height map (frontier -> 0, smooth) | min={min_val:.6g} at (x={min_xy[0]}, y={min_xy[1]})")
        else:
            axes[1].set_title("Height map (frontier -> 0, smooth)")

        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1], shrink=0.7)

        fig.tight_layout()
        save_figure(fig, "07_phase_and_height_FINAL_SMOOTH_ROI.png")
        plt.close(fig)

    if SAVE_SUMMARY_FIGURES:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(np.log1p(fft_def_mag), cmap="gray")
        axes[0].set_title("Deformed FFT magnitude (log)")
        axes[0].axis("off")
        axes[1].imshow(mask_def, cmap="gray")
        axes[1].set_title("Sideband mask used (def)")
        axes[1].axis("off")
        axes[2].imshow(np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(Idef_fft_in)))), cmap="gray")
        axes[2].set_title("FFT(I_def_fft_input) (log)")
        axes[2].axis("off")
        fig.tight_layout()
        save_figure(fig, "05_fft_debug_panels.png")
        plt.close(fig)

    if not BATCH_MODE:
        plot_height_map_interactive(height_final_out, circ_mask=roi_eroded, title="Height map (3D) - frontier->0")
        log("All figures saved. Showing interactive windows...")
        plt.show()
    log("=== FTP DEBUG RUN END ===")

    estimated_period_px = None
    try:
        vals = []
        if (k_ref is not None) and (hf_wf_ref is not None) and abs(float(k_ref[0])) > 1e-9:
            wf_ref = float(hf_wf_ref[1])
            vals.append(wf_ref / abs(float(k_ref[0])))
        if (k_def is not None) and (hf_wf_def is not None) and abs(float(k_def[0])) > 1e-9:
            wf_def = float(hf_wf_def[1])
            vals.append(wf_def / abs(float(k_def[0])))
        if len(vals) > 0:
            estimated_period_px = float(np.mean(vals))
    except Exception:
        estimated_period_px = None

    if return_results:
        out = {
            "height_map_mm_crop": height_final_out,
            "roi_eroded_crop": roi_eroded,
            "output_reliable_crop": output_reliable,
            "estimated_grating_period_px": estimated_period_px,
        }
        close_log()
        return out

    close_log()


if __name__ == "__main__":
    try:
        main()
    finally:
        close_log()

