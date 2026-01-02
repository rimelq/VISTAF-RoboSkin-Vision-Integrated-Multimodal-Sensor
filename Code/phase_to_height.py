# phase_to_height.py
# 
# Batch-calibrate: unitless FTP heightmap value -> millimeters
# 

import os
import json
import cv2
import heapq
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from skimage.restoration import unwrap_phase


# ===========================
# PATHS 
# ===========================
REFERENCE_PATH = "./Force/FINAL_reference.jpg"
DEFORMED_DIR = "./Force/Phase_to_height"  
OUTPUT_DIR   = "./Force/Phase_to_height/calibration_out"

# Initial depth, later corrected with real depth value 
# CALIBRATION_SAMPLES = [
#     ("Height_0.5mm_deformed.jpg", 0.5),
#     ("Height_1mm_deformed.jpg",   1.0),
#     ("Height_1.5mm_deformed.jpg", 1.5),
#     ("Height_2mm_deformed.jpg",   2.0),
# ]

# Real depths by forcing 0-point reference
CALIBRATION_SAMPLES = [
    ("Height_0.5mm_deformed.jpg", 1.90935),
    ("Height_1mm_deformed.jpg",   1.94770),
    ("Height_1.5mm_deformed.jpg", 2.01821),
    ("Height_2mm_deformed.jpg",   2.07255),
]


# ===========================
# PER-IMAGE HEIGHTMAP OUTPUT  
# ===========================
SAVE_HEIGHTMAP_FIGURES = True
HEIGHTMAP_FIG_DPI = 200
HEIGHTMAP_FIG_CMAP = "viridis"  
HEIGHTMAP_MARKER_COLOR = "c"    
HEIGHTMAP_OUTNAME = "heightmap.png"  
HEIGHTMAP_VMAX_FORCE_ZERO_IF_NEG = True  


# ===========================
# ROI
# ===========================
USE_FIXED_ROI = True
OUTER_CIRCLE_P1 = (1873, 1703)
OUTER_CIRCLE_P2 = (1599, 707)
OUTER_CIRCLE_P3 = (2575, 950)

ROI_ERODE_PX = 80  


# ===========================
# FTP CONFIG 
# ===========================
FFT_SIDEBAND_METHOD = "patch_shift"  

PATCH_HALF_WIDTH_BINS = 10
PATCH_WINDOW = "hann" 

BAND_RADIUS = 8
GAUSS_TRUNC_RADIUS = 24
DC_EXCLUSION = 10

FFT_PAD_PX = 96
PRE_BLUR_SIGMA_PX = 1.5

USE_CIRCULAR_APODIZATION = True
APOD_TAPER_PX = 120

ILLUM_SIGMA_PX = 45
REMOVE_MEAN_AFTER_APOD = True

# Reliability
AMP_VALID_PERCENTILE = 25.0
QUALITY_SMOOTH_SIGMA_PX = 6.0
RELIABLE_KEEP_LARGEST_CC = True
RELIABLE_EDGE_MARGIN_PX = 6
VALID_MORPH_CLOSE = True
VALID_CLOSE_KERNEL = 7
VALID_CLOSE_ITERS = 1

# Detrend
POLY_ORDER = 2
USE_TWO_PASS_DETREND = True
CONTACT_PERCENTILE = 92
DILATE_KERNEL_SIZE = 15
DILATE_ITERS = 2
MIN_CONTACT_FRAC = 0.002
MAX_CONTACT_FRAC = 0.40

# Height smoothing
RELIABLE_SMOOTH_SIGMA_PX = 2.5

# Output strategy
UNRELIABLE_BASE_VALUE = 0.0
SMOOTH_UNRELIABLE_REGION = True
UNRELIABLE_SMOOTH_SIGMA_PX = 9.0

# Frontier -> 0 
FRONTIER_ZERO_ENABLE = True
FRONTIER_ZERO_BAND_PX = 300
FRONTIER_ZERO_CURVE = "smoothstep"  # "linear" | "smoothstep" | "cosine"

# Positive deformation control
ALLOW_POSITIVE_DEFORMATION = False  # False => clamp positives to 0

# Alignment (image registration)
APPLY_GLOBAL_SHIFT = True
USE_ECC_CROP_ALIGNMENT = True
ECC_WARP_MODE = "euclidean"  # "translation" | "euclidean" | "affine"
ECC_ITERS = 300
ECC_EPS = 1e-7
ECC_GAUSS_FILT = 5

# Carrier lock + dk-ramp correction
LOCK_CARRIER_TO_REFERENCE = True
APPLY_DK_RAMP_CORRECTION = True
CARRIER_LOCAL_SEARCH_RADIUS = 6

# FFT peak selection heuristics
FORCE_RIGHT_HALF_PLANE = True
PREFER_PEAK_NEAR_CENTER_ROW = True
PEAK_MAX_DY_FROM_CENTER = 0.12

# Bad pixel handling 
BAD_PIXEL_ENABLE = True
BAD_INTENSITY_PERCENTILE = 99.9
BAD_GRADIENT_PERCENTILE = 99.7
BAD_DILATE_KSIZE = 5
BAD_DILATE_ITERS = 1
BAD_INPAINT_RADIUS = 3
BAD_INPAINT_METHOD = "telea" 

# Debug/logging
DEBUG = True


# ===========================
# CALIBRATION MODEL CONFIG  
# ===========================

APPLY_ORIGIN_CORRECTION = False

ANCHOR_ORIGIN = False
ORIGIN_WEIGHT = 20  # repeats (0,0) this many times (stronger influence)

# Fit model using x = (-min_height_unitless) so x is positive for indentations
USE_NEGATED_HEIGHT_FOR_FIT = True

# --- SciPy curve_fit models ---
FORCE_MODEL = False  # False = choose by RMSE 
FORCED_MODEL = "hinge_saturating"  # or "growth" 
MODEL_CANDIDATES = [FORCED_MODEL] if FORCE_MODEL else ["hinge_saturating", "growth"]

# curve_fit settings
CURVEFIT_MAXFEV = 200000

# Tuning for how much the exponential can bend
SAT_EXP_B_LOG10_MIN = -4.0       # smaller -> gentler; larger range -> more curvature options
SAT_EXP_B_LOG10_MAX =  4.0
SAT_EXP_B_STEPS     = 2500       # more steps -> better fit (slower)

SAT_EXP_X0_MIN_PAD_FRAC = 0.6    # x0 search from (min(x) - frac*span)
SAT_EXP_X0_MAX_PAD_FRAC = 0.2    # to (min(x) + frac*span)
SAT_EXP_X0_STEPS        = 500    # more steps -> better fit (slower)

# ===========================
# UTIL
# ===========================
def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)

def log(msg: str):
    if DEBUG:
        print(msg)

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
def circle_from_3_points(p1, p2, p3):
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    A = np.array([[2*(x2-x1), 2*(y2-y1)],
                  [2*(x3-x1), 2*(y3-y1)]], dtype=float)
    b = np.array([x2*x2 + y2*y2 - x1*x1 - y1*y1,
                  x3*x3 + y3*y3 - x1*x1 - y1*y1], dtype=float)
    cx, cy = np.linalg.solve(A, b)
    r = float(np.hypot(cx - x1, cy - y1))
    return int(round(cx)), int(round(cy)), int(round(r))

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


# ===========================
# FFT PEAKS + DEMOD
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

    if BAD_PIXEL_ENABLE:
        valid = (apo_mask > 1e-6) if apo_mask is not None else np.ones_like(img0, dtype=bool)
        bad = detect_bad_pixels(img0, valid_mask=valid)
        if np.any(bad):
            img0 = inpaint_float32(img0, bad, radius=BAD_INPAINT_RADIUS, method=BAD_INPAINT_METHOD)

    blur = cv2.GaussianBlur(img0, (0, 0), sigmaX=ILLUM_SIGMA_PX, sigmaY=ILLUM_SIGMA_PX)
    I_norm = img0 / (blur + 1e-6) - 1.0

    if PRE_BLUR_SIGMA_PX and PRE_BLUR_SIGMA_PX > 0:
        I_norm = cv2.GaussianBlur(I_norm, (0, 0), PRE_BLUR_SIGMA_PX).astype(np.float32)

    Iw = I_norm
    if apo_mask is not None:
        Iw = Iw * apo_mask

    if REMOVE_MEAN_AFTER_APOD:
        m = (apo_mask > 1e-6) if apo_mask is not None else None
        mu = _nanmedian_safe(Iw, mask=m, fallback=0.0)
        Iw = Iw - mu

    pad = int(max(0, FFT_PAD_PX))
    if pad > 0:
        Iw_fft = cv2.copyMakeBorder(Iw, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT)
    else:
        Iw_fft = Iw

    hf, wf = Iw_fft.shape
    cy, cx = hf // 2, wf // 2

    F = np.fft.fft2(Iw_fft)
    F_shift = np.fft.fftshift(F)
    fft_mag = np.abs(F_shift)

    peaks = find_top_peaks(fft_mag, dc_exclusion=dc_exclusion, n_peaks=12)

    # choose/refine carrier
    if carrier_peak is None:
        peak_x, peak_y = choose_carrier_peak(peaks, h=hf, w=wf)
        if carrier_peak_refined is None:
            peak_x_f, peak_y_f = refine_peak_parabolic_log(fft_mag, peak_x, peak_y)
        else:
            peak_x_f, peak_y_f = float(carrier_peak_refined[0]), float(carrier_peak_refined[1])
    else:
        gx, gy = float(carrier_peak[0]), float(carrier_peak[1])
        if lock_carrier_to_ref and (carrier_peak_refined is not None):
            peak_x_f, peak_y_f = float(carrier_peak_refined[0]), float(carrier_peak_refined[1])
            peak_x, peak_y = int(np.round(peak_x_f)), int(np.round(peak_y_f))
        else:
            if carrier_local_search_radius and carrier_local_search_radius > 0:
                (peak_x, peak_y), (peak_x_f, peak_y_f) = refine_peak_local_max(
                    fft_mag, gx, gy, radius=int(carrier_local_search_radius)
                )
            else:
                peak_x, peak_y = int(round(gx)), int(round(gy))
                peak_x_f, peak_y_f = refine_peak_parabolic_log(fft_mag, peak_x, peak_y)

    kx = peak_x_f - cx
    ky = peak_y_f - cy

    # sideband isolate + demod
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

        F_demod = np.fft.ifftshift(F_demod_shift)
        complex_field = np.fft.ifft2(F_demod)

        dpx = float(peak_x_f - px_i)
        dpy = float(peak_y_f - py_i)
        if abs(dpx) > 1e-6 or abs(dpy) > 1e-6:
            yy, xx = np.mgrid[0:hf, 0:wf]
            frac_demod = np.exp(-1j * 2.0 * np.pi * (dpx * (xx / wf) + dpy * (yy / hf)))
            complex_field = complex_field * frac_demod

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

        F_filt_shift = F_shift * gauss
        F_filt = np.fft.ifftshift(F_filt_shift)
        complex_field = np.fft.ifft2(F_filt)

        yy, xx = np.mgrid[0:hf, 0:wf]
        demod = np.exp(-1j * 2.0 * np.pi * (kx * (xx / wf) + ky * (yy / hf)))
        complex_demod_full = complex_field * demod

    if pad > 0:
        complex_demod = complex_demod_full[pad:pad+Iw.shape[0], pad:pad+Iw.shape[1]]
    else:
        complex_demod = complex_demod_full

    amp2 = np.abs(complex_demod).astype(np.float32)

    return (
        complex_demod,
        (int(round(peak_x_f)), int(round(peak_y_f))),   
        (float(peak_x_f), float(peak_y_f)),            
        amp2,
        (float(kx), float(ky)),
        (hf, wf),
    )


# ===========================
# QUALITY-GUIDED UNWRAP 
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

def clamp_positive_to_zero(z, mask=None):
    out = z.astype(np.float32).copy()
    if mask is None:
        m = np.isfinite(out)
    else:
        m = mask.astype(bool) & np.isfinite(out)
    out[m] = np.minimum(out[m], 0.0)
    return out


# ===========================
# FRONTIER -> 0
# ===========================
def _curve01(t, kind="smoothstep"):
    t = np.clip(t, 0.0, 1.0).astype(np.float32)
    k = str(kind).lower().strip()
    if k == "linear":
        return t
    if k == "cosine":
        return (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)

def apply_frontier_zero_transition(height, roi_mask, reliable_mask, band_px, curve="smoothstep", base_value=0.0,
                                   apply_inside=True, apply_outside=True):
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
# MAIN FTP PIPELINE (returns height_final + masks)
# ===========================
def compute_height_map_pair(reference_path: str, deformed_path: str):
    ref_bgr = cv2.imread(reference_path, cv2.IMREAD_COLOR)
    def_bgr = cv2.imread(deformed_path, cv2.IMREAD_COLOR)
    if ref_bgr is None:
        raise RuntimeError(f"Could not read reference image: {reference_path}")
    if def_bgr is None:
        raise RuntimeError(f"Could not read deformed image: {deformed_path}")
    if ref_bgr.shape != def_bgr.shape:
        raise RuntimeError("Reference and deformed images have different sizes.")

    H, W = ref_bgr.shape[:2]

    ref_gray_full = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    def_gray_full = cv2.cvtColor(def_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # global shift
    shift, response = estimate_global_shift(ref_gray_full, def_gray_full)
    if APPLY_GLOBAL_SHIFT:
        M = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], dtype=np.float32)
        def_bgr = cv2.warpAffine(def_bgr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # ROI
    if USE_FIXED_ROI:
        cx_full, cy_full, r_full = circle_from_3_points(OUTER_CIRCLE_P1, OUTER_CIRCLE_P2, OUTER_CIRCLE_P3)
    else:
        raise RuntimeError("This batch script expects USE_FIXED_ROI=True.")

    x1 = max(0, cx_full - r_full)
    x2 = min(W, cx_full + r_full)
    y1 = max(0, cy_full - r_full)
    y2 = min(H, cy_full + r_full)

    ref_crop = ref_bgr[y1:y2, x1:x2]
    def_crop = def_bgr[y1:y2, x1:x2]

    ref_gray = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)
    def_gray = cv2.cvtColor(def_crop, cv2.COLOR_BGR2GRAY)

    h, w = ref_gray.shape
    cx_local = cx_full - x1
    cy_local = cy_full - y1
    r_local = int(min(r_full, cx_local, cy_local, w - 1 - cx_local, h - 1 - cy_local))

    circ_mask = create_circular_mask(h, w, cx_local, cy_local, r_local)

    r_valid = max(0, r_local - int(ROI_ERODE_PX))
    roi_eroded = create_circular_mask(h, w, cx_local, cy_local, r_valid)

    apo = None
    if USE_CIRCULAR_APODIZATION:
        apo = create_circular_apodization(h, w, cx_local, cy_local, r_local, APOD_TAPER_PX)

    # ECC alignment on crop
    if USE_ECC_CROP_ALIGNMENT:
        aligned_def, warp, cc = align_crop_ecc(
            ref_gray, def_gray, mask_bool=circ_mask,
            mode=ECC_WARP_MODE, iters=ECC_ITERS, eps=ECC_EPS, gauss_filt=ECC_GAUSS_FILT
        )
        def_gray = aligned_def

    # FTP ref
    cref, peak_ref_int, peak_ref_f, amp_ref, k_ref, hf_wf_ref = ftp_complex_demod(
        ref_gray,
        band_radius=BAND_RADIUS,
        dc_exclusion=DC_EXCLUSION,
        carrier_peak=None,
        carrier_peak_refined=None,
        carrier_local_search_radius=0,
        apo_mask=apo,
        tag="ref",
        lock_carrier_to_ref=False
    )

    # FTP def
    if LOCK_CARRIER_TO_REFERENCE:
        cdef, peak_def_int, peak_def_f, amp_def, k_def, hf_wf_def = ftp_complex_demod(
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
        cdef, peak_def_int, peak_def_f, amp_def, k_def, hf_wf_def = ftp_complex_demod(
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
    hf, wf = hf_wf_def

    # reliability
    amp_prod = (amp_ref * amp_def).astype(np.float32)
    quality = amp_prod
    if QUALITY_SMOOTH_SIGMA_PX and QUALITY_SMOOTH_SIGMA_PX > 0:
        quality = cv2.GaussianBlur(quality, (0, 0), QUALITY_SMOOTH_SIGMA_PX).astype(np.float32)

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

    if RELIABLE_KEEP_LARGEST_CC and np.any(reliable):
        reliable = largest_connected_component(reliable) & roi_eroded

    if RELIABLE_EDGE_MARGIN_PX and RELIABLE_EDGE_MARGIN_PX > 0 and np.any(reliable):
        reliable = erode_by_distance(reliable, RELIABLE_EDGE_MARGIN_PX)

    if not np.any(reliable):
        raise RuntimeError("Reliable mask is empty (try lowering AMP_VALID_PERCENTILE or margins).")

    # phase diff
    ratio = cdef * np.conj(cref)

    if APPLY_DK_RAMP_CORRECTION and ((abs(dkx) > 1e-6) or (abs(dky) > 1e-6)):
        yy, xx = np.mgrid[0:h, 0:w]
        ramp = np.exp(1j * 2.0 * np.pi * (dkx * (xx / max(1, wf)) + dky * (yy / max(1, hf))))
        ratio = ratio * ramp

    phase_diff_wrapped = np.angle(ratio).astype(np.float32)
    phase_diff_unwrapped = unwrap_quality_guided(phase_diff_wrapped, reliable, quality=quality)

    # detrend
    if not USE_TWO_PASS_DETREND:
        _, fit = robust_polyfit2d(phase_diff_unwrapped, reliable, order=POLY_ORDER)
        phase_detrended = (phase_diff_unwrapped - fit).astype(np.float32)
        bg_med = _nanmedian_safe(phase_detrended, mask=reliable, fallback=0.0)
        phase_zeroed = phase_detrended - float(bg_med)
        contact_d = np.zeros_like(reliable, dtype=bool)
    else:
        _, fit0 = robust_polyfit2d(phase_diff_unwrapped, reliable, order=POLY_ORDER)
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
            background = reliable.copy()

        _, fit = robust_polyfit2d(phase_diff_unwrapped, background, order=POLY_ORDER)
        phase_detrended = (phase_diff_unwrapped - fit).astype(np.float32)

        bg_med = _nanmedian_safe(phase_detrended, mask=background, fallback=None)
        if bg_med is None or not np.isfinite(bg_med):
            bg_med = _nanmedian_safe(phase_detrended, mask=reliable, fallback=0.0)

        phase_zeroed = phase_detrended - float(bg_med)

    height_map = phase_zeroed.copy()

    if RELIABLE_SMOOTH_SIGMA_PX and RELIABLE_SMOOTH_SIGMA_PX > 0:
        height_map = masked_gaussian_smooth(height_map, reliable & np.isfinite(height_map), sigma=RELIABLE_SMOOTH_SIGMA_PX)

    # build final height
    height_rel = np.full((h, w), np.nan, np.float32)
    height_rel[reliable & np.isfinite(height_map)] = height_map[reliable & np.isfinite(height_map)]
    output_reliable = reliable & np.isfinite(height_rel)

    if FRONTIER_ZERO_ENABLE and FRONTIER_ZERO_BAND_PX and FRONTIER_ZERO_BAND_PX > 0:
        height_rel = apply_frontier_zero_transition(
            height_rel,
            roi_mask=roi_eroded,
            reliable_mask=output_reliable,
            band_px=FRONTIER_ZERO_BAND_PX,
            curve=FRONTIER_ZERO_CURVE,
            base_value=UNRELIABLE_BASE_VALUE,
            apply_inside=True,
            apply_outside=False
        )

    height_final = np.full((h, w), np.nan, np.float32)
    height_final[roi_eroded] = float(UNRELIABLE_BASE_VALUE)
    height_final[output_reliable] = height_rel[output_reliable]

    if SMOOTH_UNRELIABLE_REGION and UNRELIABLE_SMOOTH_SIGMA_PX and UNRELIABLE_SMOOTH_SIGMA_PX > 0:
        smooth_all = masked_gaussian_smooth(height_final, roi_eroded, sigma=UNRELIABLE_SMOOTH_SIGMA_PX)
        upd = roi_eroded & (~output_reliable)
        height_final[upd] = smooth_all[upd]

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

    if not ALLOW_POSITIVE_DEFORMATION:
        height_final = clamp_positive_to_zero(height_final, mask=roi_eroded)

    return height_final, roi_eroded, output_reliable


def compute_min_height(height_final, mask):
    m = mask.astype(bool) & np.isfinite(height_final)
    if not np.any(m):
        return np.nan, None
    tmp = np.full_like(height_final, np.inf, dtype=np.float32)
    tmp[m] = height_final[m].astype(np.float32)
    iy, ix = np.unravel_index(int(np.argmin(tmp)), tmp.shape)
    return float(height_final[iy, ix]), (int(ix), int(iy))


# ===========================
# SAVE HEIGHTMAP FIGURE 
# ===========================
def save_heightmap_figure(height_final, roi_mask, min_val, min_xy, out_path, title_prefix=""):
    ensure_output_dir(os.path.dirname(out_path))

    Z = height_final.astype(np.float32).copy()
    Z[~roi_mask.astype(bool)] = np.nan

    finite = np.isfinite(Z)
    if not np.any(finite):
        return

    vmin = float(np.nanmin(Z[finite]))
    vmax_data = float(np.nanmax(Z[finite]))
    if HEIGHTMAP_VMAX_FORCE_ZERO_IF_NEG and vmax_data <= 1e-9:
        vmax = 0.0
    else:
        vmax = vmax_data

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(Z, cmap=HEIGHTMAP_FIG_CMAP, vmin=vmin, vmax=vmax)
    ax.contour(roi_mask.astype(bool), colors="k", linewidths=0.8)

    if min_xy is not None:
        ax.scatter([min_xy[0]], [min_xy[1]], marker="x", s=140, c=HEIGHTMAP_MARKER_COLOR, linewidths=2.5)

    txt = f"min = {min_val:.6g}"
    if min_xy is not None:
        txt += f"\n(x={min_xy[0]}, y={min_xy[1]})"
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, ha="left", va="bottom", fontsize=11, color="k")

    if title_prefix:
        ax.set_title(title_prefix)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=HEIGHTMAP_FIG_DPI)
    plt.close(fig)


# ===========================
# MODEL FITTING (AUTO BEST)
# ===========================
def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)

def _aicc(n, k, sse):
    sse = float(max(sse, 1e-12))
    if n <= 0:
        return float("inf")
    aic = n * np.log(sse / n) + 2.0 * k
    if n <= (k + 1):
        return float("inf")
    return float(aic + (2.0 * k * (k + 1)) / (n - k - 1))

def _fit_linear0(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    denom = float(np.sum(x * x))
    a = float(np.sum(x * y) / denom) if denom > 1e-12 else float("nan")
    yhat = a * x
    sse = float(np.sum((y - yhat) ** 2))
    eq = f"y = {a:.6g} x"
    return {"type": "linear_through_origin", "params": {"a": a}, "k": 1, "yhat": yhat, "sse": sse, "equation": eq}

def _fit_linear(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    A = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    yhat = a * x + b
    sse = float(np.sum((y - yhat) ** 2))
    eq = f"y = {a:.6g} x + {b:.6g}"
    return {"type": "linear", "params": {"a": a, "b": b}, "k": 2, "yhat": yhat, "sse": sse, "equation": eq}

def _fit_poly2(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return None
    c2, c1, c0 = [float(v) for v in np.polyfit(x, y, deg=2)]
    yhat = c2 * x * x + c1 * x + c0
    sse = float(np.sum((y - yhat) ** 2))
    eq = f"y = {c2:.6g} x^2 + {c1:.6g} x + {c0:.6g}"
    return {"type": "poly2", "params": {"c0": c0, "c1": c1, "c2": c2}, "k": 3, "yhat": yhat, "sse": sse, "equation": eq}

def _fit_exp(x, y):
    # y = a * exp(b*x), requires y>0
    x = np.asarray(x, float); y = np.asarray(y, float)
    if np.any(y <= 0):
        return None
    ly = np.log(y)
    A = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    b, lna = float(coef[0]), float(coef[1])
    a = float(np.exp(lna))
    yhat = a * np.exp(b * x)
    sse = float(np.sum((y - yhat) ** 2))
    eq = f"y = {a:.6g} * exp({b:.6g} x)"
    return {"type": "exp", "params": {"a": a, "b": b}, "k": 2, "yhat": yhat, "sse": sse, "equation": eq}

def _fit_power(x, y):
    # y = a * x^b, requires x>0, y>0
    x = np.asarray(x, float); y = np.asarray(y, float)
    if np.any(x <= 0) or np.any(y <= 0):
        return None
    lx = np.log(x)
    ly = np.log(y)
    A = np.column_stack([lx, np.ones_like(lx)])
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    b, lna = float(coef[0]), float(coef[1])
    a = float(np.exp(lna))
    yhat = a * (x ** b)
    sse = float(np.sum((y - yhat) ** 2))
    eq = f"y = {a:.6g} * x^{b:.6g}"
    return {"type": "power", "params": {"a": a, "b": b}, "k": 2, "yhat": yhat, "sse": sse, "equation": eq}


def _fit_sat_exp(x, y, a_fixed=None):
    # y = a*(1 - exp(-b*x)), requires x>=0 and y>=0
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if np.any(x < 0) or np.any(y < 0):
        return None

    x_max = float(np.max(x))
    if x_max <= 1e-12:
        return None

    # Search b on a log grid (scaled by x-span)
    b_grid = np.logspace(-3, 3, 400) / max(1e-6, x_max)

    best = None

    if a_fixed is not None:
        a = float(a_fixed)
        for b in b_grid:
            u = 1.0 - np.exp(-b * x)
            yhat = a * u
            sse = float(np.sum((y - yhat) ** 2))
            if best is None or sse < best["sse"]:
                best = {"a": a, "b": float(b), "yhat": yhat, "sse": sse}
        if best is None:
            return None
        eq = f"y = {best['a']:.6g} * (1 - exp(-{best['b']:.6g} x))"
        # only b is effectively free if a is fixed
        return {"type": "sat_exp", "params": {"a": best["a"], "b": best["b"]}, "k": 1,
                "yhat": best["yhat"], "sse": best["sse"], "equation": eq}

    # If a not fixed: grid-search b, solve a in closed form each time
    for b in b_grid:
        u = 1.0 - np.exp(-b * x)
        denom = float(np.sum(u * u))
        if denom <= 1e-12:
            continue
        a = float(np.sum(u * y) / denom)
        yhat = a * u
        sse = float(np.sum((y - yhat) ** 2))
        if best is None or sse < best["sse"]:
            best = {"a": a, "b": float(b), "yhat": yhat, "sse": sse}

    if best is None:
        return None

    eq = f"y = {best['a']:.6g} * (1 - exp(-{best['b']:.6g} x))"
    return {"type": "sat_exp", "params": {"a": best["a"], "b": best["b"]}, "k": 2,
            "yhat": best["yhat"], "sse": best["sse"], "equation": eq}


def _fit_sat_exp_shift(x, y):
    # ORIGIN-CONSTRAINED:
    # y = a * (g(x) - g(0))
    # where g(x) = 1 - exp(-b*max(x-x0,0))
    # To have y(0)=0 exactly
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if np.any(x < 0) or np.any(y < 0):
        return None

    n = len(x)
    if n < 2:
        return None

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    span  = float(max(1e-12, x_max - x_min))

    # b grid (logspace scaled by span)
    b_grid = (10.0 ** np.linspace(SAT_EXP_B_LOG10_MIN, SAT_EXP_B_LOG10_MAX, int(SAT_EXP_B_STEPS))) / span

    # x0 grid around min(x)
    x0_lo = x_min - float(SAT_EXP_X0_MIN_PAD_FRAC) * span
    x0_hi = x_min + float(SAT_EXP_X0_MAX_PAD_FRAC) * span
    x0_grid = np.linspace(x0_lo, x0_hi, int(SAT_EXP_X0_STEPS))

    best = None

    for x0 in x0_grid:
        xeff = np.maximum(x - x0, 0.0)

        for b in b_grid:
            g  = 1.0 - np.exp(-b * xeff)

            # g(0)
            g0 = 1.0 - np.exp(-b * max(0.0 - x0, 0.0))

            u = g - g0  

            denom = float(np.sum(u * u))
            if denom <= 1e-12:
                continue

            a = float(np.sum(u * y) / denom)
            if a < 0:
                continue

            yhat = a * u
            sse = float(np.sum((y - yhat) ** 2))

            if best is None or sse < best["sse"]:
                best = {"a": a, "b": float(b), "x0": float(x0), "yhat": yhat, "sse": sse}

    if best is None:
        return None

    eq = (
        f"y = {best['a']:.6g} * ( (1 - exp(-{best['b']:.6g}*max(x-{best['x0']:.6g},0)))"
        f" - (1 - exp(-{best['b']:.6g}*max(0-{best['x0']:.6g},0))) )"
    )

    return {
        "type": "sat_exp_shift",
        "params": {"a": best["a"], "b": best["b"], "x0": best["x0"]},
        "k": 3,
        "yhat": best["yhat"],
        "sse": best["sse"],
        "equation": eq
    }

def _model_growth(x, a, b):
    # y = a*(exp(b*x) - 1)  --> y(0)=0
    return a * (np.exp(b * x) - 1.0)

def _model_hinge_sat(x, a, b, c):
    # y = a * ( (1-exp(-b*max(x-c,0))) - (1-exp(-b*max(0-c,0))) )
    return a * (
        (1.0 - np.exp(-b * np.maximum(x - c, 0.0)))
        - (1.0 - np.exp(-b * np.maximum(0.0 - c, 0.0)))
    )

def _rmse(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def _fit_growth_curvefit(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if np.any(x < 0) or np.any(y < 0):
        return None

    p0 = [max(np.max(y), 1e-6), 1.0]
    bounds = ([0.0, 0.0], [np.inf, np.inf])

    try:
        popt, _ = curve_fit(
            _model_growth, x, y, p0=p0, bounds=bounds, maxfev=int(CURVEFIT_MAXFEV)
        )
    except Exception:
        return None

    a, b = float(popt[0]), float(popt[1])
    yhat = _model_growth(x, a, b)
    sse = float(np.sum((y - yhat) ** 2))
    rmse = _rmse(y, yhat)
    eq = f"y = {a:.6g} * (exp({b:.6g} x) - 1)"
    return {"type": "growth", "params": {"a": a, "b": b}, "k": 2, "yhat": yhat, "sse": sse, "rmse": rmse, "equation": eq}

def _fit_hinge_sat_curvefit(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if np.any(x < 0) or np.any(y < 0):
        return None

    # decent defaults
    p0 = [max(np.max(y), 1e-6), 2.0, 0.0]         
    xmax = float(np.max(x)) if len(x) else 1.0
    c_lo = -0.5 * xmax
    c_hi =  1.2 * xmax
    p0 = [max(np.max(y), 1e-6), 2.0, 0.2 * xmax]
    bounds = ([0.0, 0.0, c_lo], [np.inf, np.inf, c_hi])


    try:
        popt, _ = curve_fit(
            _model_hinge_sat, x, y, p0=p0, bounds=bounds, maxfev=int(CURVEFIT_MAXFEV)
        )
    except Exception:
        return None

    a, b, c = float(popt[0]), float(popt[1]), float(popt[2])
    yhat = _model_hinge_sat(x, a, b, c)
    sse = float(np.sum((y - yhat) ** 2))
    rmse = _rmse(y, yhat)
    eq = (
        f"y = {a:.6g} * ((1-exp(-{b:.6g}*max(x-{c:.6g},0)))"
        f" - (1-exp(-{b:.6g}*max(0-{c:.6g},0))))"
    )
    return {"type": "hinge_saturating", "params": {"a": a, "b": b, "c": c},
            "k": 3, "yhat": yhat, "sse": sse, "rmse": rmse, "equation": eq}


def fit_best_model(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = int(len(x))

    candidates = []

    for name in MODEL_CANDIDATES:
        if name == "growth":
            m = _fit_growth_curvefit(x, y)
        elif name == "hinge_saturating":
            m = _fit_hinge_sat_curvefit(x, y)
        elif name == "linear0":
            m = _fit_linear0(x, y)
        elif name == "linear":
            m = _fit_linear(x, y)
        elif name == "poly2":
            m = _fit_poly2(x, y)
        elif name == "sat_exp_shift":
            m = _fit_sat_exp_shift(x, y)
        elif name == "exp":
            m = _fit_exp(x, y)
        elif name == "power":
            m = _fit_power(x, y)
        else:
            m = None

        if m is None:
            continue

        if "rmse" not in m:
            m["rmse"] = _rmse(y, m["yhat"])

        m["r2"] = r2_score(y, m["yhat"])
        candidates.append(m)


    if not candidates:
        raise RuntimeError("No valid model candidates could be fit (check your x/y values).")

    best = min(candidates, key=lambda d: d["rmse"])
    best["n"] = n
    best["x"] = x.tolist()
    best["y"] = y.tolist()
    best["candidates_summary"] = [
        {"type": c["type"], "rmse": float(c["rmse"]), "r2": float(c["r2"]), "sse": float(c["sse"])}
        for c in sorted(candidates, key=lambda d: d["rmse"])
    ]
    return best

def model_predict(model, xs):

    offset = float(model.get("origin_correction", 0.0))

    xs = np.asarray(xs, float)
    t = model["type"]
    p = model["params"]

    if t == "linear_through_origin":
        return float(p["a"]) * xs - offset
    if t == "linear":
        return (float(p["a"]) * xs + float(p["b"])) - offset
    if t == "poly2":
        c0, c1, c2 = float(p["c0"]), float(p["c1"]), float(p["c2"])
        return (c2 * xs * xs + c1 * xs + c0) - offset
    if t == "exp":
        return float(p["a"]) * np.exp(float(p["b"]) * xs) - offset
    if t == "power":
        return float(p["a"]) * (xs ** float(p["b"])) - offset
    if t == "sat_exp":
        return float(p["a"]) * (1.0 - np.exp(-float(p["b"]) * xs)) - offset
    if t == "sat_exp_shift":
        a  = float(p["a"])
        b  = float(p["b"])
        x0 = float(p["x0"])

        xeff = np.maximum(xs - x0, 0.0)
        g  = 1.0 - np.exp(-b * xeff)

        g0 = 1.0 - np.exp(-b * max(0.0 - x0, 0.0))  # scalar
        return a * (g - g0) - offset
    if t == "growth":
        a = float(p["a"]); b = float(p["b"])
        return a * (np.exp(b * xs) - 1.0) - offset
    if t == "hinge_saturating":
        a = float(p["a"]); b = float(p["b"]); c = float(p["c"])
        return a * (
            (1.0 - np.exp(-b * np.maximum(xs - c, 0.0)))
            - (1.0 - np.exp(-b * np.maximum(0.0 - c, 0.0)))
        ) - offset



    raise ValueError(f"Unknown model type: {t}")

def height_to_mm_function(model, use_negated_height=True):
    def f(h):
        h = np.asarray(h, dtype=np.float32)
        x = (-h) if use_negated_height else h
        return model_predict(model, x)
    return f


# ===========================
# RUN CALIBRATION
# ===========================
def main():
    ensure_output_dir(OUTPUT_DIR)

    rows = []
    unitless_mins = []
    depths_mm = []

    for fname, depth in CALIBRATION_SAMPLES:
        deformed_path = os.path.join(DEFORMED_DIR, fname)
        if not os.path.isfile(deformed_path):
            log(f"[SKIP] Missing file: {deformed_path}")
            continue

        stem = os.path.splitext(fname)[0]
        per_img_outdir = os.path.join(OUTPUT_DIR, stem)
        ensure_output_dir(per_img_outdir)

        log(f"\n=== Processing: {fname}  (depth={depth} mm) ===")
        height_final, roi_eroded, output_reliable = compute_height_map_pair(REFERENCE_PATH, deformed_path)

        # Use ROI 
        min_val, min_xy = compute_min_height(height_final, roi_eroded)

        # Save per-image heightmap picture 
        if SAVE_HEIGHTMAP_FIGURES:
            out_fig = os.path.join(per_img_outdir, HEIGHTMAP_OUTNAME)
            save_heightmap_figure(
                height_final=height_final,
                roi_mask=roi_eroded,
                min_val=min_val,
                min_xy=min_xy,
                out_path=out_fig,
                title_prefix=""  
            )

        rows.append({
            "file": fname,
            "depth_mm": float(depth),
            "min_height_unitless": float(min_val),
            "min_x": (min_xy[0] if min_xy else -1),
            "min_y": (min_xy[1] if min_xy else -1),
            "heightmap_figure": os.path.join(stem, HEIGHTMAP_OUTNAME) if SAVE_HEIGHTMAP_FIGURES else "",
        })

        if np.isfinite(min_val):
            unitless_mins.append(min_val)
            depths_mm.append(float(depth))

        log(f"[RESULT] min_height_unitless = {min_val:.6g} at {min_xy}  | saved: {os.path.join(per_img_outdir, HEIGHTMAP_OUTNAME)}")

    if len(unitless_mins) < 2:
        raise RuntimeError("Not enough valid samples to fit a model (need at least 2).")

    unitless_mins = np.asarray(unitless_mins, dtype=float)
    depths_mm = np.asarray(depths_mm, dtype=float)

    x = (-unitless_mins) if USE_NEGATED_HEIGHT_FOR_FIT else unitless_mins
    x = np.maximum(x, 0.0)
    y = depths_mm

    # ---- Force the fit to "see" the origin (0,0) ----
    if ANCHOR_ORIGIN:
        w = int(max(1, ORIGIN_WEIGHT))
        x = np.concatenate((np.zeros(w, dtype=float), x.astype(float)))
        y = np.concatenate((np.zeros(w, dtype=float), y.astype(float)))
    # -----------------------------------------------

    # Fit best model automatically
    best_model = fit_best_model(x, y)

    if APPLY_ORIGIN_CORRECTION:
        best_model["origin_correction"] = float(model_predict(best_model, np.array([0.0]))[0])

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "calibration_results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("file,depth_mm,min_height_unitless,min_x,min_y,heightmap_figure\n")
        for r in rows:
            f.write(f"{r['file']},{r['depth_mm']},{r['min_height_unitless']},{r['min_x']},{r['min_y']},{r['heightmap_figure']}\n")

    # Save model JSON
    model_out = {
        "reference_path": REFERENCE_PATH,
        "deformed_dir": DEFORMED_DIR,
        "output_dir": OUTPUT_DIR,
        "use_negated_height_for_fit": bool(USE_NEGATED_HEIGHT_FOR_FIT),
        "x_definition": "x = -min_height_unitless" if USE_NEGATED_HEIGHT_FOR_FIT else "x = min_height_unitless",
        "best_model": {
            "type": best_model["type"],
            "params": best_model["params"],
            "equation": best_model["equation"],
            "r2": float(best_model["r2"]),
            "rmse": float(best_model.get("rmse", np.nan)),
            "sse": float(best_model["sse"]),
            "n": int(best_model["n"]),
        },
        "candidates_summary": best_model["candidates_summary"],
        "interpretation": (
            "This model maps unitless heightmap values to mm. "
            "If use_negated_height_for_fit=true, it uses x=-height_unitless."
        ),
    }

    json_path = os.path.join(OUTPUT_DIR, "calibration_model.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(model_out, f, indent=2)

    # Plot calibration:
    fig = plt.figure(figsize=(7.2, 4.6))

    # samples 
    plt.scatter(x, y, color="C0")

    plt.scatter([0.0], [0.0], color="C0")

    xs = np.linspace(0.0, np.max(x), 400)
    plt.xlim(left=0.0)
    plt.ylim(bottom=0.0)

    ys = model_predict(best_model, xs)
    plt.plot(xs, ys)

    plt.xlabel("x = -min_height_unitless" if USE_NEGATED_HEIGHT_FOR_FIT else "x = min_height_unitless")
    plt.ylabel("depth (mm)")
    plt.title("Calibration from unitless height to mm deformation distance")  

    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "calibration_plot.png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


    log("\n=== CALIBRATION MODEL (BEST) ===")
    log(f"Saved CSV : {csv_path}")
    log(f"Saved JSON: {json_path}")
    log(f"Saved plot: {plot_path}")
    log(f"Best model: {best_model['type']}")
    log(f"Equation  : {best_model['equation']}")
    log(f"R²        : {best_model['r2']:.6g}")


if __name__ == "__main__":
    main()
