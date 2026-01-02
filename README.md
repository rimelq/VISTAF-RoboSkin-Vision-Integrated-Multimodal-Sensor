# VISTAF RoboSkin: Vision-Integrated Multimodal Sensing

A compact vision-based soft sensing skin that jointly estimates **force**, **shape**, and **temperature** from a single RGB camera view.

- **Supervisor**: Benhui Dai (CREATE Lab, EPFL)
- **Professor**: Josie Hughes (CREATE Lab, EPFL)
- **CREATE Lab** for prototyping resources

For detailed methodology, fabrication procedures, and complete calibration results, see the full project report in `docs/`.

![System Overview](docs/system_overview.png)

## Overview

VISTAF RoboSkin uses a thin (0.6 mm) silicone grating with embedded thermochromic liquid crystal (TLC) pigments. The system combines:

- **Fourier Transform Profilometry (FTP)** for deformation/shape recovery
- **CIELAB-based regression** for temperature inference from TLC color changes
- **Two-stripe design** (colored + black TLC) for both high sensitivity and broad range coverage

This enables dense depth/force and temperature maps without electronics embedded in the skin.

![Multimodal Demo](docs/multimodal_demo.png)
*Demo: A heated object pressed into the sensor reconstructs deformation (heightmap + force) and temperature (colormap).*

## How It Works

**Force/Shape Pipeline:**
1. FTP extracts phase differences between reference and deformed grating images
2. Phase → depth conversion via saturating calibration model
3. Depth integration → volume → force via exponential model

**Temperature Pipeline:**
1. CIELAB color feature extraction from TLC appearance
2. Stripe segmentation (colored vs black gratings)
3. Polynomial regression: CIELAB → Temperature

**Multimodal Integration:**
- Single camera view captures both grating deformation and TLC color
- Two stripe populations (colored: high sensitivity 20-33°C, black: broad range 10-75°C)
- Design preserves grayscale contrast for FTP while maximizing thermal information

## Performance Summary

| Module | Metric |
|--------|--------|
| Phase → Height | RMSE = 2.17 × 10⁻³ mm |
| Volume → Force | RMSE = 6.96 N |
| Temperature (Colored TLC) | RMSE = 0.44 °C |
| Temperature (Black TLC) | RMSE = 1.93 °C |

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/vistaf-roboskin.git
cd vistaf-roboskin

# Install dependencies
pip install numpy opencv-python matplotlib scikit-image scikit-learn scipy joblib
```

## Repository Structure
```
├── multimodal_sensor.py            # Main entry point - run this!
├── force_sensor.py                 # Force sensing module
├── temperature_sensor.py           # Temperature sensing module
├── shape_ftp.py                    # FTP core pipeline
├── phase_to_height.py              # Phase-to-height calibration
├── height_to_force.py              # Height-to-force calibration
├── temperature_color_model.py      # Colored TLC temperature calibration
├── temperature_black_model.py      # Black TLC temperature calibration
└── Final_demos_images/             # Demo input images
```

### Running the Multimodal Sensor

Edit the paths at the top of `multimodal_sensor.py`:
```python
REFERENCE_IMAGE = "./Final_demos_images/FINAL_reference.jpg"
DEFORMED_IMAGE  = "./Final_demos_images/FINAL_ROUND_METAL.jpg"
OUTPUT_ROOT     = "./Multimodal_Sensor/run_output"

# Optional: Show interactive 3D heightmap window
SHOW_3D_HEIGHTMAP_INTERACTIVE = True
```

```bash
python multimodal_sensor.py
```

This will:
1. Load a reference image (no contact) and a deformed image (with contact/heating)
2. Run FTP to compute shape and force
3. Run temperature inference on the deformed image
4. Generate combined outputs with visualizations
5. Save results to `./Multimodal_Sensor/run_output/session_TIMESTAMP/`

### Outputs

After running, the session folder contains:
```
session_TIMESTAMP/
├── force_sensing/
│   └── ftp_run/                    # FTP heightmaps and debug figures
├── temperature_sensing/
│   ├── temperature_map_final_colormap.png
│   └── temperature_map_final.npy
├── combined_outputs/
│   ├── multimodal_summary.json     # Complete results + calibration metrics
│   ├── force_shape_heightmap.png   # Heightmap with force annotation
│   └── temp_*.png                  # Temperature visualizations
```

The `multimodal_summary.json` includes:
- Force estimate (N), volume (cm³), contact area (mm²), max depth (mm)
- Temperature statistics (mean, median, std, min/max)
- All calibration model equations and performance metrics


## Contact

- **Rim El Qabli**: rim.elqabli@epfl.ch
- **CREATE Lab**: https://www.epfl.ch/labs/create-lab/


