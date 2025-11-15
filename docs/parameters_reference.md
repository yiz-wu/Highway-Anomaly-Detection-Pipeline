
# **Configuration File Reference (Developer Specification)**

This document explains **every field** in the pipeline configuration JSON, along with:

* What the parameter means
* Where it is used in the code
* Default values
* Accepted values / ranges
* Whether it is required

Each pipeline module reads **only the section it needs**.

---

# **Global Parameters**

| Key  | Description  | Used In  | Default | Accepted Values / Range | Required  |
| ------------- | --------------------------------------------------------------------------------------------------------- | ----------- | ------------- | ----------------------- | --------- |
| `mode`  | Defines pipeline operating mode. `"class"` runs segmentation model; `"rgb"` runs visualization-only mode. | Module 2: `script/sdf2pixelmap.py` → controls whether the predictor is `SDFPredictor` or `RGBPredictor`. | `"class"`  | `"class"`, `"rgb"`| Mandatory |
| `output_root` | Base path where all generated results will be saved. Bind a local volume to this path of container store output in your machine. | All modules | `/app/output` | Any valid path | Mandatory |


---

# **Section — `dataset`**

| Key | Meaning | Used In  | Code Locations | Default | Accepted | Required |
| ------------ | ------------------------------------------------- | ----------- | ----------------------------------- | ------- | ----------------- | -------- |
| `images`  | Path to raw image folder  | Modules 1–3 | `mapping/dataloader/sdf.py & module_1/script/interpolate_gps.py` | — | Any folder  | Yes|
| `img_format` | File extension of images  | Modules 1–2 | `mapping/dataloader/sdf.py & module_1/script/interpolate_gps.py` | `"png"` | `"png"`, `"jpg"`  | Yes |
| `car_gps` | Original or interpolated GPS file  | Modules 1–3 | `mapping/dataloader/sdf.py & module_1/script/interpolate_gps.py` | — | `.csv` or `.json` | Yes|
| `name_regex` | Regex used to extract frame indices from filename | Modules 1–2 | `mapping/dataloader/sdf.py & module_1/script/interpolate_gps.py`| `\d+`| Any regex| Optional |
| `start`| First frame index| Modules 1–3 | `mapping/dataloader/sdf.py`| — | ≥ 0| Optional |
| `end`  | Last frame index | Modules 1–3 | `mapping/dataloader/sdf.py`| — | ≥ start  | Optional |

---

# **Section — `bev`**

Defines camera calibration and BEV projection.

| Key | Meaning | Used In | Code Locations | Default | Required |
| --- | ------- | ------- | -------------- | ------- | -------- |
| `camera.name` | Camera model class name  | Module 2 | `module_2/script/sdf2pixelmap.py` | — | Yes|
| `camera.parameters` | parameters required by above camera class (usually fx, fy, cx, cy, position and orientation)| Module 2 | `module_2/script/sdf2pixelmap.py`| —  | Yes|
| `view_size`| [dist_min, dist_max, -right_max, +left_max]; distances measured in meters in the world frame, they define the portion of world ground plane to be centered on. | Module 2 | `mapping/Bev.py` | —  | Yes|
| `offset_angle`| angle in deg required to rotate the bev to look at vehicle x axis | Module 2 | `mapping/predictors/*.py` | `-90`| Optional |
| `resolution`  | Resolution in meter/pixel used to calculate the output image size (meters per pixel) | Module 2 | `mapping/Bev.py` | —  | Yes|
| `center_of_rotation`| center of rotation used to rotate the bev when generating the map | Module 2 | `mapping/predictors/*.py`| — | Optional |

---

# **Section — `model`**

| Key| Meaning| Used In  | Code Locations  | Default| Required |
| -------------- | --------------------------- | -------- | ------------------------------------ | ------------ | -------- |
| `path`| Path to custom model folder | Module 2 | `module_2/script/sdf2pixelmap.py`  | `""`| Optional |
| `name`| Model class name| Module 2 | `module_2/script/sdf2pixelmap.py` loaded via reflection  | — | Yes|
| `num_classes`  | Number of predicted classes | Module 2 | `mapping/predictors/*.py`| — | Yes|
| `parameters`| Model hyperparameters | Module 2 | `module_2/script/sdf2pixelmap.py` | — | Yes |
| `predict_size` | resized input dimensions (x,y) | Module 2 | `mapping/predictors/*.py`| `(384, 640)` | Optional |
| `checkpoint`| Path to model weights | Module 2 | `module_2/script/sdf2pixelmap.py`  | — | Optional |

---

# **Section — `pixel-mapping`**

Controls merging of frame-level predictions into a map.

| Key  | Meaning | Used In  | Code Locations | Default | Required |
| ------------------------------- | ---------------------- | -------- | ----------------------------------------- | ------- | -------- |
| `parameters.symmetric_offset`| offset in meters added to any direction. This may be used to center the chunk result or fit cropped parts| Module 2 | `mapping/PixelMap.py`  | 0 | Optional |
| `parameters.angle_mode`| orientation source  | Module 2 | `mapping/PixelMap.py`  | `ESTIMATED` | Optional |
| `parameters.split_after_frames` | chunk size | Module 2 | `mapping/PixelMap.py`  | Optional |
| `visualization.saveimages`| whether save BEV chunk images  | Module 2 | `module_2/script/sdf2pixelmap.py` | `true`  | Optional |
| `visualization.folder` | name of folder in output root where store maps | Module 2 | `module_2/script/sdf2pixelmap.py`| — | Optional  |

---

# **Section — `graph-mapping`**

Graph extraction from PixelMap.

| Key  | Meaning| Used In  | Code Locations| Default | Required |
| ------------------- | --------------------------- | -------- | ---------------------------------- | ------- | -------- |
| `start` | start frame  | Module 3 | `mapping/PixelMapIterable.py`| 0 | Optional |
| `end`| end frame (-1 = full) | Module 3 | `mapping/PixelMapIterable.py` | -1| Optional |
| `step`  | interation step to jump regions.| Module 3 | `mapping/PixelMapIterable.py` | 1 | Optional |
| `area_size`| region size in meters taken around every position of the dataloder. | Module 3 | `mapping/PixelMapIterable.py` | [20,20] | Optional |
| `ignore_classes` | List of class indexes to ignore. | Module 3 | `mapping/PixelMapIterable.py` | [] | Optional |
| `min_area_size`  | set a minimum area used to clean junk at chunk level | Module 3 | `mapping/PixelMapIterable.py`  | None | Optional |
| `parser.name` | parser class | Module 3 | `module_3/script/pixel2graph.py`  | — | Yes|
| `parser.parameters` | parser-specific params| Module 3 | depended by chosen parser | — | Yes|

---

# **Section — `postprocessing`**

Operations to clean and simplify graphs.

| Key| Meaning  | Used In  | Code Locations  | Default | Required |
| -------------------- | ----------------------------- | -------- | ------------------------------------ | ------- | -------- |
| `min_nodes` | minimum nodes to keep a graph | Module 3 | `module_3/script/pixel2graph.py`| 10| Optional |
| `stack`  | sequence of processors  | Module 3 | list of processors | — | Yes|
| `stack[].processor`  | processor class name | Module 3 | `mapping/graphs/postprocessing/*.py` | — | Yes|
| `stack[].parameters` | config for processor | Module 3 | passed to constructor | — | Optional |

---

# **Section — `lanelet2`**

Defines conversion of graph map to Lanelet2.

| Key| Meaning  | Used In  | Code Locations | Default  | Required |
| -------------------- | -------------------------------------- | -------- | -------------------------------- | -------------- | -------- |
| `input_path`| Path to GraphMap input  | Module 4 | `module_4/script/graph2let.py` | —  | Yes|
| `output_file`  | Output `.osm` file path | Module 4 | `module_4/script/graph2let.py`| —  | Yes|
| `annotated_map_path` | folder for annotated graphs| Module 4 | `module_4/script/graph2let.py` | —  | Optional |
| `zone_number`  | UTM zone number| Module 4 | `module_4/script/graph2let.py`  | 33 | Yes|
| `zone_letter`  | UTM zone letter| Module 4 | `module_4/script/graph2let.py`  | `"N"` | Yes|
| `car_height`| elevation offset  | Module 4 | `module_4/script/graph2let.py`| 0.0| Optional |
| `min_lane_size`| minimum lane width| Module 4 | `module_4/script/graph2let.py` | 1.5| Optional |
| `max_lane_size`| maximum lane width| Module 4 | `module_4/script/graph2let.py`  | 6.0| Optional |
| `search_tolerance`| tolerance when connecting lanes  | Module 4 | `module_4/script/graph2let.py`  | 0.5| Optional |
| `ignore_classes`  | classes ignored during lane conversion | Module 4 | `module_4/script/graph2let.py` | `[9,10,13,14]` | Optional |
