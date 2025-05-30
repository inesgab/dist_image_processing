# Fluo Measuring Project

## Description
This project aims to process images and the generate droplet masks, as well as tracking them over time. The tracking then enables the measurement of fluorescence in each droplet.

## Installation

### Prerequisites
- Python >= 3.10 (and < 3.13)
- Installation of sam2 (Meta, link: https://github.com/facebookresearch/sam2/tree/main. Checkpoint used (to download in the dist_image_processing repo): `sam2.1_hiera_large.pt`, also in the sam2 GitHub). Install the checkpoint in `src/`.
- Required libraries (see `requirements.txt`)

### Cell Tracking

#### Input Data Format

Data should be stored in the `data` folder:
- `data/<folder_name>/<folder_name>_t???_EGFP_ORG.tif` for fluorescence images
- `data/<folder_name>/<folder_name>_t???_DIC II 40x_ORG.tif` for DIC images

with `t???` representing the frame number (e.g., `t001`). Currently, no step > 1 is possible. The code is designed for timestamps with 3 digits for now.

#### Running the Tracking
From `src`, execute `main.py` and fill in the necessary information.

#### Output Data (Tracking)
For each input folder, there is a corresponding output folder in the `results/` directory. In `results/<folder_name>`, you will find folders each corresponding to a droplet. For example, in `results/<folder_name>/1/`, there will be three folders corresponding to droplet 1: `fluo/`, `mask/`, and `overlay/`. The latter allows you to verify if each mask is well delineated by the contours of the droplet at a given time.

#### Plotting Fluorescence
The `fluo` and `mask` folders are used for a second script. From `src`, execute `run_fluo.py`.

The resulting table and the figure of the fluorescence evolution per droplet over time are found in `results/<folder_name>/` under the names `fluorescence_results.csv` and `fluorescence_plot.png`, respectively.