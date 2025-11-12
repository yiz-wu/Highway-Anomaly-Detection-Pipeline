# Module — Data Preprocessing

## Purpose
Aligns **GPS data** with **image frames**, interpolating positions so that each image has corresponding geolocation and orientation data.  
This step ensures every image in the dataset is spatially synchronized with the vehicle’s position and attitude, preparing the data for model inference.


---

## Input
| Type | Description | Example Path | Required |
|------|--------------|---------------|-----------|
| GPS file | A `.csv` or `.json` file containing fields `sequence, x, y, z, heading, roll, pitch`. | `/app/input/gps_data/demo_gps.csv` | Mandatory |
| Image folder | Directory containing sequentially numbered camera images. | `/app/input/images/` | Mandatory |

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

These parameters are set in the JSON configuration file and control how the module reads the dataset and defines the image range.

| Parameter | Description | Default / Example | Required |
|------------|--------------|-------------------|-----------|
| `dataset.img_format` | Image format to look for (e.g., `"png"`, `"jpg"`). | `"png"` | Optional |
| `dataset.name_regex` | Regular expression pattern to extract image sequence numbers from filenames. | e.g. `r"\d+"` | Optional |
| `dataset.start` | Starting frame index to process. | `1000` | Optional |
| `dataset.end` | Ending frame index to process. | `1200` | Optional |


---

## Run Command
```bash
docker run --rm `
  -v /path/to/experiment:/app/experiment `
  1_data_preprocessing `
  -i /app/experiment/configuration.json
