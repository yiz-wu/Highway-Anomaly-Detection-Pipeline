# Module — Data Preprocessing

## Purpose
Aligns **GPS data** with **image frames**, interpolating positions so that each image has corresponding geolocation and orientation data.  
This step ensures every image in the dataset is spatially synchronized with the vehicle’s position and attitude, preparing the data for model inference.


---

## Input
| Type | Description | Required |
|------|-------------|----------|
| GPS file | A `.csv` or `.json` file containing fields `sequence, x, y, z, heading, roll, pitch`. | Mandatory |
| Image folder | Directory containing sequentially numbered camera images. | Mandatory |

---

## Processing
1. Loads GPS samples (`sequence, x, y, z, heading, roll, pitch`)
2. Loads image filenames and extracts numeric sequence indices
3. Interpolates GPS data for all image frames
4. Writes interpolated results to a CSV file

---

## Output
| File | Description | Example Path |
|------|--------------|---------------|
| `interpolated_<gps_filename>.csv` | GPS data aligned with image indices | `/app/output/interpolated_demo_gps.csv` |

---

## Parameters

These parameters are defined in the JSON configuration file and determine how the module loads the dataset and identifies the range of images to include in the interpolation.

| Parameter | Description | Default / Example | Required |
| --------- | ----------- | ----------------- | -------- |
| `dataset.images` | Path to the folder containing all camera images. The module scans this directory and loads only files matching `img_format`. | `/app/input/a1_arese` | Mandatory |
| `dataset.car_gps` | Path to the GPS log containing `sequence, x, y, z, heading, roll, pitch`. | `/app/input/gps.csv` | Mandatory |
| `dataset.img_format` | Image format/extension expected in the folder (e.g., `png`, `jpg`). Only files with this extension are processed. | `png` | Mandatory |
| `dataset.name_regex` | Regular expression used to extract the numeric frame index from image filenames (without extension). The sequence number **must be the only capture group**. | e.g., `(\d{12})_Rectified_\d+_Cam0` | Optional |
| `dataset.start` | First frame index to include. If omitted, processes starting from first frame. | `1000` | Optional |
| `dataset.end` | Last frame index to include. If omitted, processes until the last available frame. | `1200` | Optional |



---

## Run Command
```bash
docker run --rm `
  -v /path/to/experiment:/app/experiment `
  1_data_preprocessing `
  -i /app/experiment/configuration.json
