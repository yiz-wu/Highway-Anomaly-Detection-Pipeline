# Module — Graph Generation & Processing

## Purpose
Transforms the **PixelMap** generated in the previous step into a **graph-based representation** of the road network, where nodes and edges describe the spatial relationships between detected road markings and lane structures.  
Post-processing is then applied to remove noise and merge or simplify elements, improving the topological accuracy of the resulting map.


---

## Input
| Type | Description | Example Path | Required |
|------|--------------|--------------|-----------|
| JSON configuration | Configuration file defining dataset, pixel-map, graph mapping, and postprocessing parameters | `/app/input/configuration.json` | Mandatory |
| PixelMap | Output folder from Module 2 containing the BEV pixel map | `/app/output/PixelMap/` | Mandatory |
| Interpolated GPS file | `.csv` or `.json` file containing aligned GPS data (`sequence, x, y, z, heading, roll, pitch`) | `/app/input/interpolated_gps.csv` | Mandatory |
| Images folder | Directory containing dataset images | `/app/input/images/` | Mandatory |

---

## Processing
1. Loads the configuration file and dataset information.  
2. Initializes the `SDFLoader` to synchronize image and GPS data.  
3. Loads the **PixelMap** generated in the previous step.  
4. Creates a `PixelMapIterable` to iterate through BEV regions.  
5. Builds a **GraphMap** by connecting road marking elements using the selected parser.  
6. Saves the initial graph output and visualization images.  
7. Applies the postprocessing stack to clean, merge, or replace unstable detections.  
8. Exports the final postprocessed graph and saves visualization images again.

---

## Output
| File/Folder | Description | Example Path |
|--------------|-------------|--------------|
| `GraphMap/` | Raw graph structures generated from the PixelMap | `/app/output/GraphMap/0.json` |
| `GraphMapPostProcessed/` | Postprocessed graphs with cleaned connections | `/app/output/GraphMapPostProcessed/0.json` |
| `Images/GraphMap/` | Visualization of generated graphs | `/app/output/Images/GraphMap/graph_map_0_100.png` |
| `Images/GraphMapPostProcessed/` | Visualization after postprocessing | `/app/output/Images/GraphMapPostProcessed/graph_map_postprocessed_0_100.png` |

---

## Parameters
| Parameter | Description | Default / Example | Required |
|------------|-------------|-------------------|-----------|
| `dataset.images` | Path to image folder | `/app/input/a1_arese` | Mandatory |
| `dataset.car_gps` | Path to GPS data file. If an interpolated file with prefix `interpolated_` exists in the same folder, the module loads that file instead.  | `/app/input/gps.csv` | Mandatory |
| `dataset.img_format` | Image format (e.g., `"png"`, `"jpg"`) | `"png"` | Optional |
| `dataset.name_regex` | Regular expression to extract frame numbers from image filenames | `r"\d+"` | Optional |
| `dataset.start` | First frame index to process | `1050` | Optional |
| `dataset.end` | Last frame index to process | `1100` | Optional |
| `graph-mapping.start` | Starting frame index for graph extraction | `0` | Optional |
| `graph-mapping.end` | Last frame index (`-1` means until the end) | `-1` | Optional |
| `graph-mapping.step` | Step interval for processing frames | `1` | Optional |
| `graph-mapping.area_size` | Size (in meters) of the area processed at once `[width, height]` | `[20, 20]` | Mandatory |
| `graph-mapping.min_area_size` | Minimum area size for a region to be considered valid | `10000` | Optional |
| `graph-mapping.ignore_classes` | List of classes to ignore during graph construction | `[11]` | Optional |
| `graph-mapping.parser.name` | Graph parser class used to extract nodes and edges | `"BoxImageParser"` | Mandatory |
| `graph-mapping.parser.parameters` | Parameters specific to the selected parser (window height, overlap, etc.) | See example below | Mandatory |
| `postprocessing.min_nodes` | Minimum number of nodes for a valid graph | `20` | Optional |
| `postprocessing.stack` | List of post-processing processors to apply | See example below | Mandatory |
| `postprocessing.stack[].processor` | Name of processor class | `"BoxReplaceProcessor"` | Mandatory |
| `postprocessing.stack[].parameters` | Processor-specific configuration (e.g., thresholds) | `{ "min_cluster_size": 8, "replace_only": [3] }` | Optional |

---

## Run Command
```bash
docker run --rm \
  -v path_to_input:/app/input \
  -v path_to_output:/app/output \
  3_graph_generation_and_processing \
  -i configuration.json
