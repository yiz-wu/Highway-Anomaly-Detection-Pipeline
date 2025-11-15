# Module — Model Inference & BEV Merging

## Purpose

Runs a **segmentation model** on the preprocessed dataset to generate **pixel-wise predictions**, which are then projected into a **Bird’s Eye View (BEV)** map.
This step combines camera calibration data and model inference to create a spatially consistent top-down representation of detected road markings.

---

## Input

These are the real, physical data resources consumed by this module.
The JSON configuration file is intentionally **not listed here**, because it only provides parameters and instructions for processing — it is not part of the dataset itself.

| Type                         | Description                                                                                                                                                                                                                                     |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Interpolated GPS file**    | A CSV/JSON file where GPS measurements (`sequence, x, y, z, heading, roll, pitch`) have already been aligned to image frames during the preprocessing module. This provides the spatial reference for projecting segmentation outputs into BEV. |
| **Image folder**             | Directory containing the raw camera frames to be processed. Filenames must contain a numeric frame index so each image can be matched to the corresponding GPS sample.                                                                          |
| **Custom models (optional)** | A directory containing user-provided PyTorch model classes and eventually their `.pth` checkpoints. If omitted, only built-in models are available.                                                                                                 |

---

## Processing

1. Loads configuration file and extracts dataset, BEV, and model parameters.
2. Builds `SDFLoader` to read synchronized images and GPS data.
3. Constructs camera model and BEV projection from calibration parameters.
4. Loads the PyTorch segmentation model (from built-in or custom models).
5. Performs frame-by-frame segmentation and merges outputs into BEV space.
6. Saves resulting **PixelMap** and optional visual chunk images.

---

## Output

| File/Folder               | Description                                                                 | Example Path                          |
| ------------------------- | --------------------------------------------------------------------------- | ------------------------------------- |
| `PixelMap/`               | Directory containing PixelChunks (compressed BEV pixel-mapped predictions). | `/app/output/PixelMap/`               |
| `Images/PixelMap/Chunks/` | Optional visualizations of BEV map chunks in image format.                  | `/app/output/Images/PixelMap/Chunks/` |

---

## Parameters
| Parameter | Description | Default / Example | Required |
|------------|-------------|-------------------|-----------|
| `mode` | Execution mode, simply BEV project of images or also the segmentation model inference [`"RGB"`, `"class"`] | `class` | Mandatory |
| `dataset.images` | Path to image folder | `/app/input/a1_arese` | Mandatory |
| `dataset.car_gps` | Path to GPS data file. If an interpolated file with prefix `interpolated_` exists in the same folder, the module loads that file instead.  | `/app/input/gps.csv` | Mandatory |
| `dataset.img_format` | Image format (e.g., `png`, `jpg`) | `png` | | `dataset.img_format` | Image format/extension expected in the folder (e.g., `png`, `jpg`). Only files with this extension are processed.                                            | `png`                               | Mandatory |
 |
| `dataset.name_regex` | Regular expression to extract frame numbers from image filenames | `(\d{12})_Rectified_\d+_Cam0` | Optional |
| `dataset.start` | First frame index to process | `1050` | Optional |
| `dataset.end` | Last frame index to process | `1100` | Optional |
| `bev.camera.name` | Camera model class name (from `mapping.camera`) | `Camera` | Mandatory |
| `bev.camera.parameters` | Intrinsic and extrinsic camera parameters needed by specified camera class (focal lengths, offsets, orientation) | `See config example` | Mandatory |
| `bev.view_size` | Area covered by BEV in meters `[x_min, x_max, y_min, y_max]` | `[0, 10, -10, 10]` | Mandatory |
| `bev.resolution` | Pixel resolution in meters | `0.05` | Mandatory |
| `bev.offset_angle` | Rotation offset for BEV projection | `-90` | Optional |
| `bev.center_of_rotation` | Center of BEV rotation in pixel coordinates | `[200, 200]` | Optional |
| `model.path` | Directory containing model definitions | `/app/custom_models` | Optional |
| `model.name` | Model class name | `RoadStarNetE` | Mandatory |
| `model.checkpoint` | Path to pretrained model weights | `/app/input/checkpoints/ERoadNet_85_all.pth` | Optional |
| `model.num_classes` | Number of output classes predicted by the model | `12` | Mandatory |
| `model.parameters` | Model-specific hyperparameters (backbone, depth, etc.) | See config example | Mandatory |
| `pixel-mapping.parameters.symmetric_offset` | Symmetry offset for stitching multiple frames | `4` | Optional |
| `pixel-mapping.parameters.angle_mode` | Mode used to compute orientation (e.g., `gps`) | `gp"` | Optional |
| `pixel-mapping.parameters.split_after_frames` | Number of frames per processing chunk | `1000` | Optional |
| `pixel-mapping.visualization.saveimages` | Whether to save visual chunk images | `true` | Optional |
| `pixel-mapping.visualization.folder` | Folder name for image results | `Images` | Optional |

---

## Run Command
```bash
docker run --rm \
  -v path_to_custom_models:/app/custom_models \
  -v path_to_input:/app/input \
  -v path_to_output:/app/output \
  2_model_inference_bev_merging \
  -i /app/input/configuration.json
