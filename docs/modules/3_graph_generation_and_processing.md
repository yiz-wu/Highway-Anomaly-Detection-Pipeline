# Module — Graph Generation & Processing

## Purpose

Transforms the **PixelMap** generated in the previous step into a **graph-based representation** of the road network, where nodes and edges describe the spatial relationships between detected road markings and lane structures.
Post-processing is then applied to remove noise and merge or simplify elements, improving the topological accuracy of the resulting map.

---

## Input

These are the real, physical data resources consumed by this module.
The JSON configuration file is **not listed here**, because it only provides parameters — it is not part of the dataset itself.

| Type                      | Description                                                                                                                                                            |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **PixelMap folder**       | Output produced by Module 2, containing BEV pixel-mapped predictions stored as PixelChunks. This is the primary data source from which graph structures are extracted. |
| **Interpolated GPS file** | CSV/JSON file containing GPS data already aligned to image frames. Used by the graph builder to relate BEV regions to actual positions.                                |
| **Image folder**          | Directory containing the original dataset images. Filenames must include a numeric frame index for synchronization with GPS samples.                                   |

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

| File/Folder                     | Description                                      | Example Path                                                                 |
| ------------------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------- |
| `GraphMap/`                     | Raw graph structures generated from the PixelMap | `/app/output/GraphMap/0.json`                                                |
| `GraphMapPostProcessed/`        | Postprocessed graphs with cleaned connections    | `/app/output/GraphMapPostProcessed/0.json`                                   |
| `Images/GraphMap/`              | Visualization of generated graphs                | `/app/output/Images/GraphMap/graph_map_0_100.png`                            |
| `Images/GraphMapPostProcessed/` | Visualization after postprocessing               | `/app/output/Images/GraphMapPostProcessed/graph_map_postprocessed_0_100.png` |

---

## Parameters

*(unchanged — already clearly written)*

---

## Run Command

```bash
docker run --rm \
  -v path_to_input:/app/input \
  -v path_to_output:/app/output \
  3_graph_generation_and_processing \
  -i configuration.json
```

---

If you want, I can now do the same cleanup for **Module 4 — Lanelet2 Conversion**.
