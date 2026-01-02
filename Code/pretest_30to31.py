import glob
import os
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# USER CONFIGURATION
# -----------------------------

IMAGE_PATTERN = "./Pretest/30to31/pretest_30to31-*.jpg"

# Time between images in seconds
DELTA_T_SECONDS = 20

# How strict the stabilization detection is:
FRACTION_OF_TOTAL_CHANGE = 0.02  # 2% of total change
MIN_ABS_TOLERANCE = 1.0          # minimum tolerance in LAB units
CONSECUTIVE_POINTS_REQUIRED = 5  # how many consecutive points must be stable


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def sort_key_by_index(path):
    base = os.path.basename(path)
    match = re.search(r"-(\d+)\.", base)
    if match:
        return int(match.group(1))
    else:
        return base  


def load_images_sorted(pattern):
    files = glob.glob(pattern)
    if not files:
        raise RuntimeError(f"No files found for pattern: {pattern}")
    files.sort(key=sort_key_by_index)
    return files


def select_circular_roi(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots()
    ax.imshow(image_rgb)
    ax.set_title("Click center of ROI, then a point on the edge of ROI.\nClose the window when done.")
    plt.axis("off")

    print("Please click the center of the circular ROI, then a point on its edge.")
    pts = plt.ginput(2, timeout=0)  # wait until 2 clicks
    plt.close(fig)

    if len(pts) < 2:
        raise RuntimeError("ROI selection aborted: fewer than 2 points were clicked.")

    (x1, y1), (x2, y2) = pts
    radius = np.hypot(x2 - x1, y2 - y1)

    center_x = int(round(x1))
    center_y = int(round(y1))
    radius = int(round(radius))

    print(f"Selected ROI center: ({center_x}, {center_y}), radius: {radius} pixels.")
    return center_x, center_y, radius


def create_circular_mask(h, w, center_x, center_y, radius):
    Y, X = np.ogrid[:h, :w]
    dist_sq = (X - center_x) ** 2 + (Y - center_y) ** 2
    mask = dist_sq <= radius ** 2
    return mask


def detect_stabilization_time(values, times,
                              frac_change=FRACTION_OF_TOTAL_CHANGE,
                              min_abs_tol=MIN_ABS_TOLERANCE,
                              n_consecutive=CONSECUTIVE_POINTS_REQUIRED):
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)

    if len(values) == 0:
        return None

    # final value: mean of last up to 5 points
    if len(values) >= 5:
        final_value = values[-5:].mean()
    else:
        final_value = values[-1]

    total_change = final_value - values[0]
    tolerance = max(abs(total_change) * frac_change, min_abs_tol)

    print(f"Final value (approx): {final_value:.3f}")
    print(f"Total change: {total_change:.3f}")
    print(f"Stabilization tolerance: ±{tolerance:.3f} (LAB units)")

    n = len(values)
    if n < n_consecutive:
        print("Not enough points to check stabilization.")
        return None

    for i in range(n - n_consecutive + 1):
        window = values[i : i + n_consecutive]
        if np.all(np.abs(window - final_value) <= tolerance):
            return times[i]

    return None


# -----------------------------
# MAIN ANALYSIS
# -----------------------------

def main():
    image_files = load_images_sorted(IMAGE_PATTERN)
    print(f"Found {len(image_files)} images.")

    first_img = cv2.imread(image_files[0], cv2.IMREAD_COLOR)
    if first_img is None:
        raise RuntimeError(f"Could not read image: {image_files[0]}")

    center_x, center_y, radius = select_circular_roi(first_img)
    h, w = first_img.shape[:2]
    mask = create_circular_mask(h, w, center_x, center_y, radius)

    # Compute mean LAB-L inside ROI for each image
    roi_values = []
    times = []

    for idx, path in enumerate(image_files):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: skipping unreadable image: {path}")
            continue

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L_channel = lab[:, :, 0].astype(np.float32)

        roi_L = L_channel[mask]
        mean_L = float(roi_L.mean())

        t = idx * DELTA_T_SECONDS  # seconds from start

        roi_values.append(mean_L)
        times.append(t)

    roi_values = np.array(roi_values, dtype=float)
    times = np.array(times, dtype=float)

    # Detect stabilization time
    stabilization_time = detect_stabilization_time(roi_values, times)

    if stabilization_time is not None:
        print(f"Estimated stabilization time: {stabilization_time:.1f} seconds "
              f"({stabilization_time/60:.2f} minutes)")
    else:
        print("No clear stabilization time found with current parameters.")

    # Plot the evolution and mark stabilization
    plt.figure(figsize=(8, 4))
    plt.plot(times, roi_values, "o-", label="ROI mean L (LAB)")
    plt.xlabel("Time [s]")
    plt.ylabel("Mean L (LAB) in ROI")
    plt.title(f"Pre-test evolution: {IMAGE_PATTERN}")

    if len(roi_values) >= 5:
        final_val = roi_values[-5:].mean()
    else:
        final_val = roi_values[-1]
    plt.axhline(final_val, color="gray", linestyle="--", label="Final mean")

    if stabilization_time is not None:
        plt.axvline(stabilization_time, color="red", linestyle="--", label="Stabilization time")
        plt.text(stabilization_time, final_val,
                 f"  t ≈ {stabilization_time/60:.1f} min",
                 color="red", va="bottom")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
