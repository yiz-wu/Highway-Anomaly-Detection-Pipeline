
# Configuration Parameter Reference

This document details the `configuration.json` file required to run the pipeline.

**Important Path Note:**
All paths defined here must be **Container Paths**, not Host paths.

  * Reference: See [README - Understanding Path Mapping](../README.md) for details.

-----

## 1\. Global Parameters

These parameters apply to the pipeline execution environment.

```json
{
  "mode": "class",
  "output_root": "/app/output"
}
```

| Key | Description | Default | Required |
| :--- | :--- | :--- | :--- |
| `mode` | `"class"`: Runs full segmentation model.<br>`"RGB"`: Visualization-only mode (skips inference). | `"class"` | **Yes** |
| `output_root` | The destination folder inside the container. Ensure this path is mounted to a volume. | `/app/output` | **Yes** |


-----

## 2\. Dataset (`dataset`)

Defines the input data source.

```json
"dataset": {
  "images": "/app/input/images",
  "car_gps": "/app/input/gps_data.csv",
  "img_format": "png",
  "name_regex": "(\\d{12})_Rectified_\\d+_Cam0",
  "start": 0,
  "end": 2000
}
```

| Key | Description | Accepted Values | Default | Required |
| :--- | :--- | :--- | :--- | :--- |
| `images` | Directory containing raw frames. | Container Path | — | **Yes** |
| `car_gps` | File containing GPS/Pose data. | `.csv`, `.json` | — | **Yes** |
| `img_format` | Image file extension. | `"png"`, `"jpg"` | `"png"` | **Yes** |
| `name_regex` | Regex to extract frame index from filenames.<br>The sequence shall be matched by first group. | Valid Regex | `\d+` | No |
| `start` | First frame index to process. | Integer $\ge 0$ | 0 | No |
| `end` | Last frame index to process. | Integer $>$ start | -1 (All) | No |

-----

## 3\. BEV Projection (`bev`)

Defines how camera images are projected onto the Bird's Eye View (BEV) plane.

```json
"bev": {
  "resolution": 0.05,
  "view_size": [0.0, 50.0, -15.0, 15.0],
  "offset_angle": -90,
  "camera": {
    "name": "Camera",
    "parameters": { "cameraData": {
      "intrinsic": {"fx":350.975, "fy":350.975, "cx":335.952, "cy":194.081},
      "extrinsic": {"x": 0, "y": 0.06, "z": 1.55, "yaw": 0, "pitch": 0.1125, "roll":0}}}
  },
  "center_of_rotation": [200, 200]
}
```

| Key | Description | Default | Required |
| :--- | :--- | :--- | :--- |
| `resolution` | Meters per pixel in the output map. | — | **Yes** |
| `view_size` | The ground plane bounds relative to the car: `[min_dist, max_dist, right_bound, left_bound]` (in meters). | — | **Yes** |
| `offset_angle` | Rotation angle (degrees) to align the BEV with the vehicle X-axis. | `-90` | No |
| `center_of_rotation`| Specific pivot point for rotation (if not vehicle center). | — | No |
| `camera.name` | The Python class name for the camera model. | — | **Yes** |
| `camera.parameters` | Dictionary of parameters (`fx`, `fy`, `cx`, `cy`, etc.) required by the chosen camera class. | — | **Yes** |

-----

## 4\. Segmentation Model (`model`)

Configuration for the PyTorch segmentation model.

```json
"model": {
  "path": "",
  "name": "RoadStarNetE",
  "num_classes": 12,
  "parameters": {
      "line_classes": 11,
      "area_classes": 2,
      "backbone":"efficientnet-b3",
      "bifpn_activation": false,
      "decoder_depth": 64,
      "num_bifpn": 6
  },
  "predict_size": [384, 640],
  "checkpoint": "/app/input/checkpoints/ERoadNet_85_all.pth"
}
```

| Key | Description | Default | Required |
| :--- | :--- | :--- | :--- |
| `name` | Model class name (dynamically loaded). | — | **Yes** |
| `num_classes` | Total number of semantic classes the model predicts. | — | **Yes** |
| `parameters` | Hyperparameters passed to the model constructor. | — | **Yes** |
| `checkpoint` | Path to the `.pth` weight file. | — | No |
| `path` | Path to a custom model folder (if not using built-ins). | `""` | No |
| `predict_size` | Resizes input before inference `[height, width]`. | `[384, 640]` | No |

-----

## 5\. Pixel Mapping (`pixel-mapping`)

Controls how individual frames are stitched into the global pixel map.

```json
"pixel-mapping": {
  "parameters": {
    "angle_mode": "gps",
    "symmetric_offset": 4,
    "split_after_frames": 1000
  },
  "visualization": {
    "saveimages": true,
    "folder": "Images"
  }
}
```

| Key | Description | Default | Required |
| :--- | :--- | :--- | :--- |
| `parameters.symmetric_offset` | Offset (meters) added to directions (used to center chunks or fit crops). | 0 | No |
| `parameters.angle_mode` | Orientation source. `"ESTIMATED"` (model) or `"gps"`. | `"ESTIMATED"` | No |
| `parameters.split_after_frames`| Force a map chunk split after N frames. | None | No |
| `visualization.saveimages` | If `true`, saves debug images of map chunks. | `true` | No |
| `visualization.folder` | Subfolder name in `output_root` to store visualization images. | — | No |

-----

## 6\. Graph Generation (`graph-mapping`)

Settings for extracting the graph topology from the Pixel Map.

```json
"graph-mapping": {
  "start": 0,
  "end": -1,
  "step": 1,
  "area_size": [20, 20],
  "ignore_classes": [11],
  "min_area_size": 10000,
  "parser": {
    "name": "BoxImageParser",
    "parameters": {
        "window_height": 32,
        "overlap_pixels": 16,
        "resolution": 0.05
    }
  }
}
```

| Key | Description | Default | Required |
| :--- | :--- | :--- | :--- |
| `start` | Frame index to start processing. | 0 | No |
| `end` | Frame index to stop processing (-1 = end). | -1 | No |
| `step` | Iteration step size to jump between regions. | 1 | No |
| `area_size` | Size of sliding window (in meters) `[x, y]`. | `[20, 20]` | No |
| `ignore_classes` | List of class IDs to exclude from the graph. | `[]` | No |
| `min_area_size` | Minimum pixel area to consider a valid region (noise filter). | None | No |
| `parser.name` | Name of the algorithm class used to skeletonize the map. | — | **Yes** |
| `parser.parameters` | Dictionary of specific settings for the chosen parser. | — | **Yes** |

-----

## 7\. Post-Processing (`postprocessing`)

A stack of operations to clean the generated graph.

```json
"postprocessing": {
  "min_nodes": 10,
  "stack": [
    {
      "processor": "BoxReplaceProcessor",
      "parameters": { "min_cluster_size": 8, "replace_only": [3] }
    },
    {
      "processor": "SimplifyGraph",
      "parameters": { "epsilon": 2.0 }
    }
  ]
}
```

| Key | Description | Default | Required |
| :--- | :--- | :--- | :--- |
| `min_nodes` | Graphs with fewer nodes than this will be discarded. | 10 | No |
| `stack` | An ordered list of processor objects to run sequentially. | — | **Yes** |
| `stack[].processor` | The class name of the processor to run. | — | **Yes** |
| `stack[].parameters` | Configuration dictionary for that specific processor. | — | No |

-----

## 8\. Lanelet2 Conversion (`lanelet2`)

Converts the internal graph format to OpenStreetMap (OSM) Lanelet2 format.

```json
"lanelet2": {
  "input_path": "/app/output/GraphMap/final_graph.pkl",
  "output_file": "/app/output/Lanenet_output/map.osm",
  "annotated_map_path": "/app/output/Lanenet_output/",
  "zone_number": 33,
  "zone_letter": "N",
  "car_height": 0.0,
  "min_lane_size": 1.5,
  "max_lane_size": 6.0,
  "search_tolerance": 0.5,
  "ignore_classes": [9, 10, 13, 14]
}
```

| Key | Description | Default | Required |
| :--- | :--- | :--- | :--- |
| `input_path` | Path to the processed graph (`.pkl`). | — | **Yes** |
| `output_file` | Destination path for the `.osm` file. | — | **Yes** |
| `annotated_map_path` | Folder to save annotated graph visualizations. | — | No |
| `zone_number` | UTM Zone Number for georeferencing. | 33 | **Yes** |
| `zone_letter` | UTM Zone Letter. | `"N"` | **Yes** |
| `car_height` | Elevation offset (meters) for the map. | 0.0 | No |
| `min_lane_size` | Minimum lane width (meters). | 1.5 | No |
| `max_lane_size` | Maximum lane width (meters). | 6.0 | No |
| `search_tolerance` | Tolerance distance when connecting lanes. | 0.5 | No |
| `ignore_classes` | List of class IDs ignored during lane conversion. | `[9,10,13,14]` | No |
