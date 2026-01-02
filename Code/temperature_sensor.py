# temperature_sensor.py
#
# Periodic grating segmentation + temperature inference with TWO models:
#   - WIDE model  (L,a,b,gray): robust broad-range baseline
#   - COLOR model (L,a,b):      more precise but only valid in ~20..33°C range


from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  

try:
    import joblib 
except Exception as e:
    raise RuntimeError("joblib is required to load the trained models.") from e


# ============================================================
# USER CONFIG
# ============================================================

INPUT_IMAGE_PATH = "./Final_demos_images/FINAL_TEMP_DEMO.jpg"
OUTPUT_DIR = "./Temperature/temp_inference"

# ROI (outer circle) defined by 3 points (pixel coords in the input image)
OUTER_CIRCLE_P1 = (1845,1818)
OUTER_CIRCLE_P2 = (1517,623)
OUTER_CIRCLE_P3 = (2687,914)

USE_INNER_CIRCLE = False
INNER_CIRCLE_P1 = (1881, 1749)
INNER_CIRCLE_P2 = (1579, 665)
INNER_CIRCLE_P3 = (2616, 936)

# Output cropping toggle (crop ALL saved images to OUTER ROI bbox + stats over OUTER ROI)
CROP_OUTPUT_TO_OUTER_ROI = True
CROP_PAD_PX = 10  # padding around the outer ROI bounding box (pixels)

# Feature smoothing (odd; 1 disables)
BLUR_KSIZE = 5

# COLOR model valid range (expected)
COLOR_T_MIN = 20.0
COLOR_T_MAX = 33.0

# Fusion parameters
COLOR_GUARD_BAND = 0.5         # allow slight tolerance beyond [min,max]
SWITCH_MARGIN_C = 1.0          # blending width around COLOR_T_MAX

# Final clamp for outputs
FINAL_T_MIN = 20.0
FINAL_T_MAX = 75.0

# Periodic stripe segmentation via FFT sideband
SEG_BAND_RADIUS = 22
SEG_DC_EXCLUSION = 28
SEG_FORCE_RIGHT_HALF_PLANE = True
SEG_PREFER_PEAK_NEAR_CENTER_ROW = True
SEG_PEAK_MAX_DY_FROM_CENTER = 0.14  # fraction of height
SEG_ILLUM_SIGMA = 20  # illumination normalization sigma (0 disables)

# Exclude LEDs/specular highlights
SAT_THRESH_GRAY = 245
SAT_DILATE_KSIZE = 13

# Morphology to stabilize masks
POST_CLOSE_KX = 3
POST_CLOSE_KY = 31
POST_OPEN_KX = 3
POST_OPEN_KY = 7

# COLOR gating by chroma (prevents color model being used on black/gray areas)
# LAB in OpenCV: a,b are centered at 128
COLOR_CHROMA_MIN = 10.0          # increase if color model still leaks; decrease if too restrictive
COLOR_SUPPORT_DILATE = 3         # dilate light mask a little to keep continuity in "lighter stripe zones"

# Colormap style
COLORMAP_NAME = "jet"
COLORMAP_ALPHA_OVERLAY = 0.55

# Optional final smoothing (removes residual stripe texture in the temperature field)
FINAL_SMOOTH_ENABLE = True
FINAL_SMOOTH_SIGMA_ACROSS = 6.0  # pixels; across stripes direction (set 0 to disable)
FINAL_SMOOTH_SIGMA_ALONG = 1.0   # pixels; along stripes direction

DEBUG_SAVE = True


# ============================================================
# MODEL PATHS (auto-locate Temperature/)
# ============================================================

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def find_temperature_dir(start_dir: str, max_up: int = 8) -> str:
    cur = os.path.abspath(start_dir)
    for _ in range(max_up + 1):
        cand = os.path.join(cur, "Temperature")
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    raise RuntimeError(
        "Could not locate a 'Temperature' folder by walking upward from:\n"
        f"  {start_dir}\n"
        "Fix your directory structure or hardcode TEMPERATURE_DIR."
    )


TEMPERATURE_DIR = find_temperature_dir(_THIS_DIR)


def resolve_latest_joblib(pattern: str) -> str:
    matches = glob.glob(pattern)
    if not matches:
        parent_dir = os.path.dirname(pattern)
        existing = sorted(glob.glob(os.path.join(parent_dir, "*")))
        raise RuntimeError(
            f"No model matches pattern:\n  {pattern}\n\n"
            f"Directory exists? {os.path.isdir(parent_dir)}\n"
            f"Contents of {parent_dir}:\n  " + "\n  ".join(existing[:80]) +
            ("" if len(existing) <= 80 else "\n  ...")
        )
    return max(matches, key=os.path.getmtime)


COLOR_MODEL_JOBLIB = resolve_latest_joblib(os.path.join(
    TEMPERATURE_DIR, "Colored_Model", "calibration_out",
    "color_model_global_huber_deg*.joblib"
))

WIDE_MODEL_JOBLIB = resolve_latest_joblib(os.path.join(
    TEMPERATURE_DIR, "MixedColorBlack_Model", "calibration_out",
    "black_model_global_huber_deg*.joblib"
))


# ============================================================
# Geometry helpers
# ============================================================

def circle_from_three_points(p1, p2, p3, eps: float = 1e-12) -> Tuple[float, float, float]:
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
    r = float(np.hypot(x1 - cx, y1 - cy))
    return float(cx), float(cy), r


def roi_mask_from_circle(h: int, w: int, p1, p2, p3) -> np.ndarray:
    cx, cy, r = circle_from_three_points(p1, p2, p3)
    Y, X = np.ogrid[:h, :w]
    dist2 = (X - cx) ** 2 + (Y - cy) ** 2
    return (dist2 <= (r**2))


def annulus_mask(h: int, w: int,
                 inner_p1, inner_p2, inner_p3,
                 outer_p1, outer_p2, outer_p3) -> np.ndarray:
    outer = roi_mask_from_circle(h, w, outer_p1, outer_p2, outer_p3)
    inner = roi_mask_from_circle(h, w, inner_p1, inner_p2, inner_p3)
    return outer & (~inner)


def bbox_from_mask(mask: np.ndarray, pad: int = 0) -> Tuple[int, int, int, int]:
    """
    Returns (y0, y1, x0, x1) where y1/x1 are exclusive (suitable for slicing).
    If mask is empty, returns full image bbox.
    """
    h, w = mask.shape[:2]
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return 0, h, 0, w
    y0 = int(max(0, ys.min() - int(pad)))
    y1 = int(min(h, ys.max() + int(pad) + 1))
    x0 = int(max(0, xs.min() - int(pad)))
    x1 = int(min(w, xs.max() + int(pad) + 1))
    return y0, y1, x0, x1


def crop2d(arr: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    if bbox is None:
        return arr
    y0, y1, x0, x1 = bbox
    return arr[y0:y1, x0:x1]


def crop_bgr(img_bgr: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    if bbox is None:
        return img_bgr
    y0, y1, x0, x1 = bbox
    return img_bgr[y0:y1, x0:x1, :]


# ============================================================
# Model wrapper
# ============================================================

@dataclass
class TempModel:
    name: str
    pipeline: Any
    feature_names: Tuple[str, ...]
    calibrator: Optional[Any] = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = self.pipeline.predict(X).astype(np.float32)
        if self.calibrator is not None:
            pred = self.calibrator.predict(pred).astype(np.float32)
        return pred


def load_model_joblib(path: str, default_name: str) -> TempModel:
    if not os.path.exists(path):
        raise RuntimeError(f"Model file not found:\n  {path}")

    obj = joblib.load(path)

    if isinstance(obj, dict) and "model" in obj:
        pipeline = obj["model"]
        feat = obj.get("use_features", None)
        if feat is None:
            raise RuntimeError(f"Joblib at {path} missing 'use_features'.")
        calibrator = obj.get("isotonic_calibrator", None)
        name = obj.get("name", default_name)
        return TempModel(
            name=str(name),
            pipeline=pipeline,
            feature_names=tuple(feat),
            calibrator=calibrator,
        )

    raise RuntimeError(f"Unrecognized joblib format: {path}")


# ============================================================
# Feature planes
# ============================================================

def _ensure_odd(ksize: int) -> int:
    k = int(ksize)
    if k <= 1:
        return 1
    if k % 2 == 0:
        k += 1
    return k


def compute_feature_planes(image_bgr: np.ndarray, blur_ksize: int = 5) -> Dict[str, np.ndarray]:
    blur_ksize = _ensure_odd(blur_ksize)
    if blur_ksize > 1:
        image_bgr = cv2.GaussianBlur(image_bgr, (blur_ksize, blur_ksize), 0)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    return {
        "L": lab[:, :, 0],
        "a": lab[:, :, 1],
        "b": lab[:, :, 2],
        "gray": gray,
    }


def predict_map_for_mask(model: TempModel, planes: Dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.full((h, w), np.nan, dtype=np.float32)
    if not np.any(mask):
        return out

    cols = []
    for fn in model.feature_names:
        if fn not in planes:
            raise RuntimeError(f"Missing required feature plane '{fn}' for model '{model.name}'.")
        cols.append(planes[fn][mask])

    X = np.stack(cols, axis=1).astype(np.float32)
    pred = model.predict(X)
    out[mask] = pred
    return out


# ============================================================
# Periodic stripe segmentation via FFT sideband
# ============================================================

def _find_top_peaks(mag: np.ndarray, dc_exclusion: int, n_peaks: int = 12) -> list[tuple[int, int, float]]:
    mag = np.asarray(mag, float)
    h, w = mag.shape
    cy, cx = h // 2, w // 2

    ms = mag.copy()
    y0 = max(0, cy - dc_exclusion)
    y1 = min(h, cy + dc_exclusion)
    x0 = max(0, cx - dc_exclusion)
    x1 = min(w, cx + dc_exclusion)
    ms[y0:y1, x0:x1] = 0.0

    flat = ms.ravel()
    n_peaks = int(min(n_peaks, flat.size))
    idx = np.argpartition(flat, -n_peaks)[-n_peaks:]
    idx = idx[np.argsort(flat[idx])[::-1]]

    peaks: list[tuple[int, int, float]] = []
    for i in idx:
        y, x = np.unravel_index(i, ms.shape)
        peaks.append((int(x), int(y), float(ms[y, x])))
    return peaks


def _choose_carrier_peak(peaks: list[tuple[int, int, float]], h: int, w: int) -> tuple[int, int]:
    cy, cx = h // 2, w // 2
    candidates = peaks[:]

    if SEG_FORCE_RIGHT_HALF_PLANE:
        right = [p for p in candidates if p[0] > cx]
        if right:
            candidates = right

    if SEG_PREFER_PEAK_NEAR_CENTER_ROW:
        max_dy = int(SEG_PEAK_MAX_DY_FROM_CENTER * h)
        near = [p for p in candidates if abs(p[1] - cy) <= max_dy]
        if near:
            candidates = near

    if not candidates:
        candidates = peaks

    px, py, _ = max(candidates, key=lambda t: t[2])
    return int(px), int(py)


def _illum_normalize(gray_f: np.ndarray, roi: np.ndarray, sigma: int) -> np.ndarray:
    g = gray_f.astype(np.float32)
    if sigma is None or int(sigma) <= 0:
        mu = float(np.mean(g[roi])) if np.any(roi) else float(np.mean(g))
        mu = mu if abs(mu) > 1e-9 else 1.0
        return (g / mu).astype(np.float32)

    blur = cv2.GaussianBlur(g, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    blur[blur < 1e-6] = 1.0
    norm = g / blur
    mu = float(np.mean(norm[roi])) if np.any(roi) else float(np.mean(norm))
    mu = mu if abs(mu) > 1e-9 else 1.0
    return (norm / mu).astype(np.float32)


def _make_saturation_mask(image_bgr: np.ndarray, roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sat = (gray >= int(SAT_THRESH_GRAY)) & roi
    k = _ensure_odd(SAT_DILATE_KSIZE)
    if k > 1 and np.any(sat):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        sat_u8 = (sat.astype(np.uint8) * 255)
        sat_u8 = cv2.dilate(sat_u8, kernel, iterations=1)
        sat = (sat_u8 > 127) & roi
    return sat


def _postprocess_mask(m: np.ndarray, roi: np.ndarray) -> np.ndarray:
    if not np.any(m):
        return m
    close_kx = _ensure_odd(max(1, int(POST_CLOSE_KX)))
    close_ky = _ensure_odd(max(1, int(POST_CLOSE_KY)))
    open_kx = _ensure_odd(max(1, int(POST_OPEN_KX)))
    open_ky = _ensure_odd(max(1, int(POST_OPEN_KY)))

    mu8 = (m.astype(np.uint8) * 255)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kx, close_ky))
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (open_kx, open_ky))

    mu8 = cv2.morphologyEx(mu8, cv2.MORPH_CLOSE, k_close)
    mu8 = cv2.morphologyEx(mu8, cv2.MORPH_OPEN, k_open)
    out = (mu8 > 127) & roi
    return out


def _save_debug_fft_mag(fft_mag: np.ndarray, peak_xy: tuple[int, int], out_path: str,
                        crop_bbox: Optional[Tuple[int, int, int, int]] = None) -> None:
    mag = np.log1p(np.asarray(fft_mag, float))
    mag = mag - mag.min()
    if mag.max() > 1e-9:
        mag = mag / mag.max()
    img = (mag * 255.0).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    px, py = peak_xy
    cv2.drawMarker(img, (px, py), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    img = crop_bgr(img, crop_bbox)
    cv2.imwrite(out_path, img)


def _save_seg_overlay(gray_u8: np.ndarray, roi_full: np.ndarray, dark: np.ndarray, out_path: str,
                      crop_bbox: Optional[Tuple[int, int, int, int]] = None) -> None:
    base = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    light = roi_full & (~dark)
    overlay[light] = (0, 255, 0)  # green
    overlay[dark] = (0, 0, 255)   # red
    alpha = 0.45
    out = base.copy()
    sel = roi_full
    out[sel] = cv2.addWeighted(base[sel], 1.0 - alpha, overlay[sel], alpha, 0.0)
    out = crop_bgr(out, crop_bbox)
    cv2.imwrite(out_path, out)


def segment_dark_light_gratings_periodic_fft(image_bgr: np.ndarray, roi_full: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    sat = _make_saturation_mask(image_bgr, roi_full)
    roi_eff = roi_full & (~sat)
    if not np.any(roi_eff):
        raise RuntimeError("ROI became empty after saturation exclusion. Lower SAT_THRESH_GRAY / dilation.")

    g = gray.copy()
    med = float(np.median(g[roi_eff]))
    g[~roi_full] = med

    I_norm = _illum_normalize(g, roi_eff, SEG_ILLUM_SIGMA)

    F = np.fft.fft2(I_norm)
    F_shift = np.fft.fftshift(F)
    mag = np.abs(F_shift)

    peaks = _find_top_peaks(mag, dc_exclusion=int(SEG_DC_EXCLUSION), n_peaks=16)
    if not peaks:
        raise RuntimeError("Could not find FFT peaks for stripe carrier.")
    peak_x, peak_y = _choose_carrier_peak(peaks, h=h, w=w)

    Y, X = np.ogrid[:h, :w]
    dist2 = (X - peak_x) ** 2 + (Y - peak_y) ** 2
    bp = dist2 <= (float(SEG_BAND_RADIUS) ** 2)

    F_filt_shift = F_shift * bp
    z = np.fft.ifft2(np.fft.ifftshift(F_filt_shift))  # complex carrier field

    # Rotate so real part aligns with stripe modulation
    m = (I_norm - 1.0).astype(np.float32)
    c = np.sum(z[roi_eff] * m[roi_eff])
    phi0 = float(np.angle(c)) if np.isfinite(c) else 0.0
    z_rot = z * np.exp(-1j * phi0)

    s = np.real(z_rot).astype(np.float32)
    mask_a = (s >= 0) & roi_eff
    mask_b = (s < 0) & roi_eff

    mean_a = float(np.mean(gray[mask_a])) if np.any(mask_a) else 1e9
    mean_b = float(np.mean(gray[mask_b])) if np.any(mask_b) else 1e9

    if mean_a <= mean_b:
        dark = mask_a
        light = mask_b
        chosen = "A_is_dark"
    else:
        dark = mask_b
        light = mask_a
        chosen = "B_is_dark"

    dark = _postprocess_mask(dark, roi_eff)
    light = _postprocess_mask(light, roi_eff)

    dark_final = dark & roi_eff
    light_final = roi_eff & (~dark_final)

    # stripe frequency vector (direction of variation across stripes)
    cy, cx = h // 2, w // 2
    dx = float(peak_x - cx)
    dy = float(peak_y - cy)
    fx = dx / float(w)
    fy = dy / float(h)
    fmag = float(np.hypot(fx, fy))
    period_px = (1.0 / fmag) if fmag > 1e-9 else float("nan")
    angle_rad = float(np.arctan2(dy, dx))  

    dbg = {
        "peak_x": int(peak_x),
        "peak_y": int(peak_y),
        "phi0_rad": float(phi0),
        "mean_gray_A": float(mean_a),
        "mean_gray_B": float(mean_b),
        "chosen": chosen,
        "roi_pixels": int(np.count_nonzero(roi_full)),
        "roi_eff_pixels": int(np.count_nonzero(roi_eff)),
        "sat_pixels": int(np.count_nonzero(sat)),
        "dark_pixels": int(np.count_nonzero(dark_final)),
        "light_pixels": int(np.count_nonzero(light_final)),
        "carrier_angle_rad": angle_rad,
        "carrier_period_px": float(period_px),
    }
    pack = {
        "dbg": dbg,
        "fft_mag": mag,
        "signal": s,
        "roi_eff": roi_eff,
        "sat": sat,
        "peak": (peak_x, peak_y),
        "angle_rad": angle_rad,
        "period_px": period_px,
    }
    return dark_final, light_final, pack


# ============================================================
# Map utilities
# ============================================================

def clamp_map(m: np.ndarray, roi: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = m.copy()
    sel = roi & np.isfinite(out)
    out[sel] = np.clip(out[sel], float(lo), float(hi))
    out[~roi] = np.nan
    return out


def inpaint_temperature_map(temp_map: np.ndarray, roi_mask: np.ndarray, radius: int = 7) -> np.ndarray:
    """
    Inpaint only inside roi_mask. Outside roi_mask becomes NaN.
    Missing = inside roi_mask but not finite.
    """
    out = temp_map.copy()
    inside = roi_mask
    known = inside & np.isfinite(out)
    missing = inside & (~np.isfinite(out))

    if not np.any(missing) or not np.any(known):
        out[~inside] = np.nan
        return out

    vals = out[known]
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if vmax - vmin < 1e-6:
        out[missing] = vmin
        out[~inside] = np.nan
        return out

    scaled = np.zeros_like(out, dtype=np.uint8)
    scaled[known] = ((out[known] - vmin) / (vmax - vmin) * 255.0).clip(0, 255).astype(np.uint8)
    mask_inpaint = (missing.astype(np.uint8) * 255)

    filled = cv2.inpaint(scaled, mask_inpaint, inpaintRadius=int(radius), flags=cv2.INPAINT_TELEA)

    out_filled = out.copy()
    out_filled[inside] = (filled[inside].astype(np.float32) / 255.0) * (vmax - vmin) + vmin
    out_filled[~inside] = np.nan
    return out_filled


def dilate_bool_mask(m: np.ndarray, k: int) -> np.ndarray:
    k = _ensure_odd(int(k))
    if k <= 1 or not np.any(m):
        return m
    mu8 = (m.astype(np.uint8) * 255)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    d = cv2.dilate(mu8, ker, iterations=1)
    return d > 127


# ============================================================
# Fusion logic
# ============================================================

def fuse_maps_per_pixel(
    roi: np.ndarray,
    wide_map: np.ndarray,
    color_map: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    final = wide everywhere
    if color is valid near [COLOR_T_MIN, COLOR_T_MAX] -> use it
    blend near COLOR_T_MAX to keep continuity
    """
    final = wide_map.copy()
    source = np.zeros(final.shape, dtype=np.uint8)  # 0=wide, 255=color, 128=blend

    wide_ok = roi & np.isfinite(wide_map)
    color_ok = (
        roi
        & np.isfinite(color_map)
        & (color_map >= (COLOR_T_MIN - COLOR_GUARD_BAND))
        & (color_map <= (COLOR_T_MAX + COLOR_GUARD_BAND))
    )

    final[color_ok] = color_map[color_ok]
    source[color_ok] = 255

    low_th = COLOR_T_MAX - SWITCH_MARGIN_C
    high_th = COLOR_T_MAX + SWITCH_MARGIN_C
    blend_zone = wide_ok & color_ok & (wide_map > low_th) & (wide_map < high_th)
    if np.any(blend_zone):
        w = (high_th - wide_map[blend_zone]) / (high_th - low_th)
        w = np.clip(w, 0.0, 1.0).astype(np.float32)
        final[blend_zone] = w * color_map[blend_zone] + (1.0 - w) * wide_map[blend_zone]
        source[blend_zone] = 128

    final = clamp_map(final, roi, FINAL_T_MIN, FINAL_T_MAX)

    dbg = {
        "roi_pixels": int(np.count_nonzero(roi)),
        "wide_ok_pixels": int(np.count_nonzero(wide_ok)),
        "color_ok_pixels": int(np.count_nonzero(color_ok)),
        "blend_pixels": int(np.count_nonzero(blend_zone)),
    }
    return final.astype(np.float32), source, dbg


# ============================================================
# Visualization with legend
# ============================================================

def save_colormap_with_legend(temp_map: np.ndarray, roi: np.ndarray, out_path: str,
                             vmin: float, vmax: float, cmap: str = "jet",
                             title: Optional[str] = None) -> None:
    m = temp_map.copy()
    m[~roi] = np.nan
    fig = plt.figure(figsize=(8, 8), dpi=200)
    ax = fig.add_subplot(111)
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=20)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Temperature (°C)")
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_colormap_overlay_with_legend(image_bgr: np.ndarray, temp_map: np.ndarray, roi: np.ndarray,
                                      out_path: str, vmin: float, vmax: float,
                                      cmap: str = "jet", alpha: float = 0.55,
                                      title: Optional[str] = None) -> None:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)
    m = temp_map.copy()
    m[~roi] = np.nan

    fig = plt.figure(figsize=(8, 8), dpi=200)
    ax = fig.add_subplot(111)
    ax.imshow(rgb)
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, alpha=float(alpha))
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Temperature (°C)")
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_horizontal_legend_only(out_path: str, vmin: float, vmax: float, cmap: str = "jet") -> None:
    """
    Saves a HORIZONTAL colorbar-only legend image (no data).
    """
    from matplotlib import cm, colors  # local import

    fig = plt.figure(figsize=(10, 1.2), dpi=200)
    cax = fig.add_axes([0.06, 0.55, 0.88, 0.25])

    norm = colors.Normalize(vmin=float(vmin), vmax=float(vmax))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Temperature (°C)")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ============================================================
# Optional final smoothing (map-domain)
# ============================================================

def oriented_gaussian_blur_float(map_f: np.ndarray, roi: np.ndarray,
                                 angle_rad: float,
                                 sigma_across: float,
                                 sigma_along: float) -> np.ndarray:
    """
    Rotate map so the "across stripes" direction aligns with +x,
    apply anisotropic GaussianBlur, rotate back.
    """
    if sigma_across <= 0 and sigma_along <= 0:
        out = map_f.copy()
        out[~roi] = np.nan
        return out

    h, w = map_f.shape
    center = (w / 2.0, h / 2.0)
    angle_deg = -float(angle_rad) * 180.0 / float(np.pi)

    map0 = map_f.copy()
    map0[~np.isfinite(map0)] = 0.0
    roi_u8 = (roi.astype(np.uint8) * 255)

    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rot_map = cv2.warpAffine(map0, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    rot_roi = cv2.warpAffine(roi_u8, M, (w, h), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 127

    sx = float(max(0.0, sigma_across))
    sy = float(max(0.0, sigma_along))
    blurred = cv2.GaussianBlur(rot_map.astype(np.float32), (0, 0), sigmaX=sx, sigmaY=sy)

    M_inv = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
    back = cv2.warpAffine(blurred, M_inv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    back_roi = cv2.warpAffine((rot_roi.astype(np.uint8) * 255), M_inv, (w, h),
                              flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 127

    out = back.astype(np.float32)
    out[~back_roi] = np.nan
    return out


# ============================================================
# Main
# ============================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_IMAGE_PATH):
        raise RuntimeError(f"INPUT_IMAGE_PATH not found:\n  {os.path.abspath(INPUT_IMAGE_PATH)}")

    img = cv2.imread(INPUT_IMAGE_PATH, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {INPUT_IMAGE_PATH}")

    h, w = img.shape[:2]

    roi_outer = roi_mask_from_circle(h, w, OUTER_CIRCLE_P1, OUTER_CIRCLE_P2, OUTER_CIRCLE_P3)

    if USE_INNER_CIRCLE:
        roi_full = annulus_mask(h, w,
                                INNER_CIRCLE_P1, INNER_CIRCLE_P2, INNER_CIRCLE_P3,
                                OUTER_CIRCLE_P1, OUTER_CIRCLE_P2, OUTER_CIRCLE_P3)
    else:
        roi_full = roi_outer

    crop_bbox = bbox_from_mask(roi_outer, pad=int(CROP_PAD_PX)) if bool(CROP_OUTPUT_TO_OUTER_ROI) else None


    roi_vis_and_stats = roi_outer if bool(CROP_OUTPUT_TO_OUTER_ROI) else roi_full

    print("Resolved COLOR_MODEL_JOBLIB:", COLOR_MODEL_JOBLIB)
    print("Resolved WIDE_MODEL_JOBLIB :", WIDE_MODEL_JOBLIB)

    color_model = load_model_joblib(COLOR_MODEL_JOBLIB, "color_model")
    wide_model = load_model_joblib(WIDE_MODEL_JOBLIB, "wide_model")

    if tuple(color_model.feature_names) != ("L", "a", "b"):
        raise RuntimeError(f"Color model must use ('L','a','b'), got {color_model.feature_names}")
    if tuple(wide_model.feature_names) != ("L", "a", "b", "gray"):
        raise RuntimeError(f"Wide model must use ('L','a','b','gray'), got {wide_model.feature_names}")

    # --- Segment periodic stripes (dark vs light) ---
    dark_mask, light_mask, seg_pack = segment_dark_light_gratings_periodic_fft(img, roi_full)
    roi_eff = seg_pack["roi_eff"]  
    sat = seg_pack["sat"]

    # --- Compute LAB + chroma for color gating ---
    planes = compute_feature_planes(img, blur_ksize=BLUR_KSIZE)
    a = planes["a"]
    b = planes["b"]
    chroma = np.sqrt((a - 128.0) ** 2 + (b - 128.0) ** 2).astype(np.float32)

    # --- Build COLOR support mask ---
    light_d = dilate_bool_mask(light_mask, COLOR_SUPPORT_DILATE)
    color_support = light_d & roi_eff & (~sat) & (chroma >= float(COLOR_CHROMA_MIN))

    cv2.imwrite(os.path.join(OUTPUT_DIR, "mask_roi.png"),
                (crop2d(roi_full, crop_bbox).astype(np.uint8) * 255))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "mask_roi_eff.png"),
                (crop2d(roi_eff, crop_bbox).astype(np.uint8) * 255))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "mask_sat.png"),
                (crop2d(sat, crop_bbox).astype(np.uint8) * 255))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "mask_dark.png"),
                (crop2d(dark_mask, crop_bbox).astype(np.uint8) * 255))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "mask_light.png"),
                (crop2d(light_mask, crop_bbox).astype(np.uint8) * 255))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "mask_color_support.png"),
                (crop2d(color_support, crop_bbox).astype(np.uint8) * 255))

    if DEBUG_SAVE:
        gray_u8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _save_seg_overlay(gray_u8, roi_full, dark_mask,
                          os.path.join(OUTPUT_DIR, "debug_seg_overlay.png"),
                          crop_bbox=crop_bbox)
        _save_debug_fft_mag(seg_pack["fft_mag"], seg_pack["peak"],
                            os.path.join(OUTPUT_DIR, "debug_fft_mag.png"),
                            crop_bbox=crop_bbox)

        ch = chroma.copy()
        ch[~roi_full] = 0
        denom = (np.nanpercentile(ch[roi_full], 99) + 1e-6) if np.any(roi_full) else 1.0
        ch_u8 = np.clip((ch / denom) * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "debug_chroma_u8.png"), crop2d(ch_u8, crop_bbox))

    # --- Predict WIDE everywhere (baseline) ---
    wide_map_raw = predict_map_for_mask(wide_model, planes, roi_eff)

    # --- Predict COLOR only where we truly have colorful light gratings ---
    color_map_raw = predict_map_for_mask(color_model, planes, color_support)

    # --- Inpaint ONLY to fill small holes inside their intended domains ---
    wide_map = inpaint_temperature_map(wide_map_raw, roi_full, radius=7)
    wide_map = clamp_map(wide_map, roi_full, FINAL_T_MIN, FINAL_T_MAX)

    color_map = inpaint_temperature_map(color_map_raw, color_support, radius=5)
    color_map = clamp_map(color_map, color_support, COLOR_T_MIN - 5.0, COLOR_T_MAX + 5.0)

    # --- Fuse per pixel ---
    final_fused, source_map, fuse_dbg = fuse_maps_per_pixel(roi_full, wide_map, color_map)

    # Save color_ok mask explicitly
    color_ok_mask = (roi_full & np.isfinite(color_map) &
                     (color_map >= (COLOR_T_MIN - COLOR_GUARD_BAND)) &
                     (color_map <= (COLOR_T_MAX + COLOR_GUARD_BAND)))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "mask_color_ok.png"),
                (crop2d(color_ok_mask, crop_bbox).astype(np.uint8) * 255))

    # --- Optional final smoothing ---
    if FINAL_SMOOTH_ENABLE:
        angle = float(seg_pack.get("angle_rad", 0.0))
        final_map = oriented_gaussian_blur_float(final_fused, roi_full,
                                                 angle_rad=angle,
                                                 sigma_across=float(FINAL_SMOOTH_SIGMA_ACROSS),
                                                 sigma_along=float(FINAL_SMOOTH_SIGMA_ALONG))
        final_map = clamp_map(final_map, roi_full, FINAL_T_MIN, FINAL_T_MAX)
    else:
        final_map = final_fused

    # Min/max in stats ROI
    inside = roi_vis_and_stats & np.isfinite(final_map)
    max_temp = float(np.nanmax(final_map[inside])) if np.any(inside) else float("nan")
    min_temp = float(np.nanmin(final_map[inside])) if np.any(inside) else float("nan")

    # --- Save arrays (always full-size) ---
    np.save(os.path.join(OUTPUT_DIR, "temperature_map_fused.npy"), final_fused.astype(np.float32))
    np.save(os.path.join(OUTPUT_DIR, "temperature_map_final.npy"), final_map.astype(np.float32))

    # --- Save debug maps (optionally cropped) ---
    wide_map_raw_dark = predict_map_for_mask(wide_model, planes, dark_mask & roi_eff)
    save_colormap_with_legend(
        crop2d(wide_map_raw_dark, crop_bbox),
        crop2d(roi_vis_and_stats, crop_bbox),
        os.path.join(OUTPUT_DIR, "debug_wide_raw_dark_only_colormap.png"),
        vmin=FINAL_T_MIN, vmax=FINAL_T_MAX, cmap=COLORMAP_NAME
    )

    wide_map_raw_light = predict_map_for_mask(wide_model, planes, light_mask & roi_eff)
    save_colormap_with_legend(
        crop2d(wide_map_raw_light, crop_bbox),
        crop2d(roi_vis_and_stats, crop_bbox),
        os.path.join(OUTPUT_DIR, "debug_wide_raw_light_only_colormap.png"),
        vmin=FINAL_T_MIN, vmax=FINAL_T_MAX, cmap=COLORMAP_NAME
    )

    save_colormap_with_legend(
        crop2d(wide_map_raw, crop_bbox),
        crop2d(roi_vis_and_stats, crop_bbox),
        os.path.join(OUTPUT_DIR, "debug_wide_raw_colormap.png"),
        vmin=FINAL_T_MIN, vmax=FINAL_T_MAX, cmap=COLORMAP_NAME
    )

    save_colormap_with_legend(
        crop2d(color_map_raw, crop_bbox),
        crop2d(roi_vis_and_stats, crop_bbox),
        os.path.join(OUTPUT_DIR, "debug_color_raw_colormap_20_33.png"),
        vmin=COLOR_T_MIN, vmax=COLOR_T_MAX, cmap=COLORMAP_NAME
    )

    save_colormap_with_legend(
        crop2d(color_map_raw, crop_bbox),
        crop2d(roi_vis_and_stats, crop_bbox),
        os.path.join(OUTPUT_DIR, "debug_color_raw_colormap_20_75.png"),
        vmin=FINAL_T_MIN, vmax=FINAL_T_MAX, cmap=COLORMAP_NAME
    )

    cv2.imwrite(os.path.join(OUTPUT_DIR, "debug_fused_source.png"), crop2d(source_map, crop_bbox))

    # --- Save fused visualizations ---
    save_colormap_with_legend(
        crop2d(final_fused, crop_bbox),
        crop2d(roi_vis_and_stats, crop_bbox),
        os.path.join(OUTPUT_DIR, "temperature_map_fused_colormap.png"),
        vmin=FINAL_T_MIN, vmax=FINAL_T_MAX, cmap=COLORMAP_NAME
    )

    save_colormap_overlay_with_legend(
        crop_bgr(img, crop_bbox),
        crop2d(final_fused, crop_bbox),
        crop2d(roi_vis_and_stats, crop_bbox),
        os.path.join(OUTPUT_DIR, "temperature_map_fused_colormap_overlay.png"),
        vmin=FINAL_T_MIN, vmax=FINAL_T_MAX, cmap=COLORMAP_NAME, alpha=COLORMAP_ALPHA_OVERLAY
    )

    # --- Save final visualizations (final result) ---
    title_final = f"Temperature map - min: {min_temp:.2f} °C, max: {max_temp:.2f} °C"
    save_colormap_with_legend(
        crop2d(final_map, crop_bbox),
        crop2d(roi_vis_and_stats, crop_bbox),
        os.path.join(OUTPUT_DIR, "temperature_map_final_colormap.png"),
        vmin=FINAL_T_MIN, vmax=FINAL_T_MAX, cmap=COLORMAP_NAME,
        title=title_final
    )

    save_colormap_overlay_with_legend(
        crop_bgr(img, crop_bbox),
        crop2d(final_map, crop_bbox),
        crop2d(roi_vis_and_stats, crop_bbox),
        os.path.join(OUTPUT_DIR, "temperature_map_final_colormap_overlay.png"),
        vmin=FINAL_T_MIN, vmax=FINAL_T_MAX, cmap=COLORMAP_NAME, alpha=COLORMAP_ALPHA_OVERLAY
    )

    # Save horizontal legend-only image (final scale)
    save_horizontal_legend_only(
        os.path.join(OUTPUT_DIR, "temperature_legend_horizontal.png"),
        vmin=FINAL_T_MIN, vmax=FINAL_T_MAX, cmap=COLORMAP_NAME
    )

    roi_name = "OUTER ROI" if bool(CROP_OUTPUT_TO_OUTER_ROI) else "ROI"
    print(f"Final temperature min/max in {roi_name}: {min_temp:.3f} / {max_temp:.3f} °C")
    print("Segmentation debug:")
    for k in sorted(seg_pack["dbg"].keys()):
        print(f"  {k}: {seg_pack['dbg'][k]}")
    print("Fusion debug:")
    for k in sorted(fuse_dbg.keys()):
        print(f"  {k}: {fuse_dbg[k]}")
    print(f"Saved outputs to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()

