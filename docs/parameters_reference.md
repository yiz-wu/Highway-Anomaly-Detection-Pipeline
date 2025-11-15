
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

| Key           | Description                                                                                               | Used In     | Default       | Accepted Values / Range | Required  |
| ------------- | --------------------------------------------------------------------------------------------------------- | ----------- | ------------- | ----------------------- | --------- |
| `mode`        | Defines pipeline operating mode. `"class"` runs segmentation model; `"rgb"` runs visualization-only mode. | Module 2: `script/sdf2pixelmap.py` → controls whether the predictor is `SDFPredictor` or `RGBPredictor`.    | `"class"`     | `"class"`, `"rgb"`      | Mandatory |
| `output_root` | Base path where all generated results will be saved. Bind a local volume to this path of container store output in your machine.          | All modules | `/app/output` | Any valid path          | Mandatory |


---

# **Section — `dataset`**

| Key          | Meaning                                           | Used In     | Code Locations                      | Default | Accepted          | Required |
| ------------ | ------------------------------------------------- | ----------- | ----------------------------------- | ------- | ----------------- | -------- |
| `images`     | Path to raw image folder                          | Modules 1–3 | `mapping/dataloader.py → SDFLoader` | —       | Any folder        | Yes      |
| `img_format` | File extension of images                          | Modules 1–2 | `SDFLoader.__init__`                | `"png"` | `"png"`, `"jpg"`  | Optional |
| `car_gps`    | Original or interpolated GPS file                 | Modules 1–3 | `script/*` loaders → `SDFLoader`    | —       | `.csv` or `.json` | Yes      |
| `name_regex` | Regex used to extract frame indices from filename | Modules 1–2 | `SDFLoader._extract_idx`            | `\d+`   | Any regex         | Optional |
| `start`      | First frame index                                 | Modules 1–3 | `SDFLoader.restart_from_()`         | —       | ≥ 0               | Optional |
| `end`        | Last frame index                                  | Modules 1–3 | `SDFLoader.end_at_()`               | —       | ≥ start           | Optional |

---

# **Section — `bev`**

Defines camera calibration and BEV projection.

| Key                                      | Meaning                                       | Used In  | Code Locations                                      | Default     | Required |
| ---------------------------------------- | --------------------------------------------- | -------- | --------------------------------------------------- | ----------- | -------- |
| `camera.name`                            | Camera model class name                       | Module 2 | `mapping/camera/*.py` loaded via `name2class_map()` | `"Camera"`  | Yes      |
| `camera.parameters.cameraData.intrinsic` | fx, fy, cx, cy parameters                     | Module 2 | `mapping/camera/Camera.py → build_intrinsic()`      | —           | Yes      |
| `camera.parameters.cameraData.extrinsic` | camera position and orientation               | Module 2 | `Camera.__init__()`                                 | —           | Yes      |
| `view_size`                              | BEV span in meters `[xmin, xmax, ymin, ymax]` | Module 2 | `mapping/Bev.py → Bev.__init__()`                   | —           | Yes      |
| `offset_angle`                           | Rotation offset applied to BEV                | Module 2 | `SDFPredictor.__init__()`                           | `0`         | Optional |
| `resolution`                             | BEV resolution (meters per pixel)             | Module 2 | `Bev.__init__()`                                    | —           | Yes      |
| `center_of_rotation`                     | BEV rotation pivot (pixel coords)             | Module 2 | `SDFPredictor.__init__()`                           | `[200,200]` | Optional |

---

# **Section — `model`**

| Key            | Meaning                     | Used In  | Code Locations                       | Default      | Required |
| -------------- | --------------------------- | -------- | ------------------------------------ | ------------ | -------- |
| `path`         | Path to custom model folder | Module 2 | `models/load_custom_models()`        | `""`         | Optional |
| `name`         | Model class name            | Module 2 | `models/*.py` loaded via reflection  | —            | Yes      |
| `num_classes`  | Number of predicted classes | Module 2 | `SDFPredictor.__init__()`            | —            | Yes      |
| `parameters`   | Model hyperparameters       | Module 2 | passed directly to model constructor | —            | Optional |
| `predict_size` | resized input dims          | Module 2 | `SDFPredictor.predict_at`            | `[384, 640]` | Optional |
| `checkpoint`   | Path to model weights       | Module 2 | `torch.load()`                       | —            | Optional |

---

# **Section — `pixel-mapping`**

Controls merging of frame-level predictions into a map.

| Key                             | Meaning                | Used In  | Code Locations                            | Default | Accepted        | Required |
| ------------------------------- | ---------------------- | -------- | ----------------------------------------- | ------- | --------------- | -------- |
| `parameters.symmetric_offset`   | merging offset         | Module 2 | `PixelMap.from_dataset()`                 | 4       | ≥ 0             | Optional |
| `parameters.angle_mode`         | orientation source     | Module 2 | `PixelMap.from_dataset()`                 | `"gps"` | `"gps"`,`"imu"` | Optional |
| `parameters.split_after_frames` | chunk size             | Module 2 | `PixelMap.from_dataset()`                 | 1000    | ≥ 1             | Optional |
| `visualization.saveimages`      | save BEV chunk images  | Module 2 | `sdf2pixelmap.py → map.save_chunks_rgb()` | `true`  | true/false      | Optional |
| `visualization.folder`          | folder name for images | Module 2 | `Images/`                                 | —       | Optional        |          |

---

# **Section — `graph-mapping`**

Graph extraction from PixelMap.

| Key                 | Meaning                     | Used In  | Code Locations                     | Default | Required |
| ------------------- | --------------------------- | -------- | ---------------------------------- | ------- | -------- |
| `start`             | start frame                 | Module 3 | `PixelMapIterable.__iter__()`      | 0       | Optional |
| `end`               | end frame (-1 = full)       | Module 3 | `pixel2graph.py`                   | -1      | Optional |
| `step`              | step between frames         | Module 3 | `pixel2graph.py`                   | 1       | Optional |
| `area_size`         | tile size (m)               | Module 3 | `PixelMapIterable(region_size)`    | —       | Yes      |
| `ignore_classes`    | classes excluded from graph | Module 3 | `PixelMapIterable(ignore_classes)` | —       | Optional |
| `min_area_size`     | minimum area accepted       | Module 3 | `PixelMapIterable(min_area_size)`  | —       | Optional |
| `parser.name`       | parser class                | Module 3 | `mapping/graphs/*.py`              | —       | Yes      |
| `parser.parameters` | parser-specific params      | Module 3 | depended by chosen parser          | —       | Yes      |

---

# **Section — `postprocessing`**

Operations to clean and simplify graphs.

| Key                  | Meaning                       | Used In  | Code Locations                       | Default | Required |
| -------------------- | ----------------------------- | -------- | ------------------------------------ | ------- | -------- |
| `min_nodes`          | minimum nodes to keep a graph | Module 3 | `GraphMap.nx_graphs(min_nodes)`      | 10      | Optional |
| `stack`              | sequence of processors        | Module 3 | list of processors                   | —       | Yes      |
| `stack[].processor`  | processor class name          | Module 3 | `mapping/graphs/postprocessing/*.py` | —       | Yes      |
| `stack[].parameters` | config for processor          | Module 3 | passed to constructor                | —       | Optional |

---

# **Section — `lanelet2`**

Defines conversion of graph map to Lanelet2.

| Key                  | Meaning                                | Used In  | Code Locations                   | Default        | Required |
| -------------------- | -------------------------------------- | -------- | -------------------------------- | -------------- | -------- |
| `input_path`         | Path to GraphMap input                 | Module 4 | `graph2lanelet2.py → map.load()` | —              | Yes      |
| `output_file`        | Output `.osm` file path                | Module 4 | `lanelet2.io.write()`            | —              | Yes      |
| `annotated_map_path` | folder for annotated graphs            | Module 4 | optional                         | —              | Optional |
| `zone_number`        | UTM zone number                        | Module 4 | used in `UtmProjector`           | 33             | Yes      |
| `zone_letter`        | UTM zone letter                        | Module 4 | used in `UtmProjector`           | `"N"`          | Yes      |
| `car_height`         | elevation offset                       | Module 4 | affects `elevated_pos()`         | 0.0            | Optional |
| `min_lane_size`      | minimum lane width                     | Module 4 | passed to `LinesSplitPropagator` | 1.5            | Optional |
| `max_lane_size`      | maximum lane width                     | Module 4 | `LinesSplitPropagator`           | 6.0            | Optional |
| `search_tolerance`   | tolerance when connecting lanes        | Module 4 | `LinesSplitPropagator`           | 0.5            | Optional |
| `ignore_classes`     | classes ignored during lane conversion | Module 4 | used in filtering linestrings    | `[9,10,13,14]` | Optional |

---

# If you want…

I can also generate:

✔ **Direct hyperlinks** (relative GitHub links) to each script or class
✔ A **compact one-page cheatsheet** version
✔ A **validation schema** (JSON Schema) that can automatically validate configuration files
✔ A **template configuration file** with comments
✔ Integration with VS Code autocompletion (JSON schema association)

Just tell me what you prefer.
