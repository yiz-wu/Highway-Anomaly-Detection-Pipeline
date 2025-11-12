# A Working Example

This guide walks through the **full execution of the 4 pipeline modules** using the dataset of a1_arese **[downloadable from OneDrive [link](https://www.dropbox.com/scl/fo/y5sw9ob8jjacvnwi7o9j5/ADgXFKlWGZRhwavQNieBwbs?rlkey=lt200bvpot5akey7gr141owfn&st=fo358wi2&dl=0), not included here because too big]**.  
It covers everything from building the Docker images to running each container step-by-step.

---

## 1. Build All Docker Images

Before running any module, make sure all the container images are built and available locally.  
You can follow the detailed explanation in [docker_build.md](../docs/docker_build.md), and here’s a quick recap:

```bash
docker build -t 1_data_preprocessing module_1_preprocessing/
docker build -t 2_model_inference_bev_merging module_2_model_inference/
docker build -t 3_graph_generation_and_processing module_3_graph_generation/
docker build -t lanelet2 Lanelet2/
docker build -t 4_lanelet2_conversion module_4_lanelet2_conversion/
```

---

## 2. Dataset Folder Structure

The demo dataset is organized as follows:

```
DEMO_MATERIAL/
│
├── a1_arese/                 ← shall contain all camera images
│
├── checkpoints/              ← folder containing model weights
│   └── ERoadNet_85_all.pth
│
├── custom_model/             ← optional directory for custom PyTorch model test
│   ├── deeplabv3_resnet50_coco-cd0a2569.pth
│   └── TorchvisionModel.py
│
├── gps.csv                   ← raw GPS data (sequence, x, y, z, heading, roll, pitch)
│
└── demo_a1_arese.json        ← configuration file used by all modules
```


* All **input data** (images, GPS, models) are under a single directory.
* Each module can mount the same folder as `/app/input` inside its container.
* The same JSON configuration file (`demo_a1_arese.json`) is reused for all pipeline stages.
* The generated results (interpolated GPS, pixel maps, graphs, and final Lanelet2 maps) will be automatically written inside the container’s `/app/output`, which maps to your host `DEMO_MATERIAL/output/` (created at runtime).
* **Recover missing checkpoint weight of model and dataset a1_arese**
---

## 3. Run the Pipeline Step-by-Step

All modules share the same JSON configuration file and use consistent mounting points (except the last module):

* `/app/input` → mapped to your `DEMO_MATERIAL/` folder
* `/app/output` → mapped to `DEMO_MATERIAL/output/` on your host machine


---

### Step 1 — Data Preprocessing

This step aligns each image frame with its corresponding GPS position by interpolation.

```bash
docker run --rm \
  -v $(pwd)/DEMO_MATERIAL:/app/input \
  -v $(pwd)/DEMO_MATERIAL/output:/app/output \
  1_data_preprocessing \
  -i /app/input/demo_a1_arese.json
```

**Output:**

```
/app/input/interpolated_gps.csv
```

This file will be used automatically in the next step if detected.

---

### Step 2 — Model Inference & BEV Merging

This module runs the segmentation model on each frame and merges predictions into a BEV pixel map.


```bash
docker run --rm \
  -v $(pwd)/DEMO_MATERIAL:/app/input \
  -v $(pwd)/DEMO_MATERIAL/output:/app/output \
  -v $(pwd)/DEMO_MATERIAL/custom_model:/app/custom_models \
  2_model_inference_bev_merging \
  -i /app/input/demo_a1_arese.json
```

**Output directories:**

```
/app/output/PixelMap/
/app/output/Images/PixelMap/
```

These contain both pixel-level map data and visual BEV chunk images.

To test the custom model, change the model section of the configuration file to 
```json
"model": {
    "path": "/app/custom_models",
    "name": "TorchvisionModel",
    "parameters": {
      "pretrained": true,
      "num_classes": 12,
      "checkpoint_path": "/app/custom_models/deeplabv3_resnet50_coco-cd0a2569.pth"
    },
    "num_classes": 12,
    "predict_size": [384, 640]
},
```

---

### Step 3 — Graph Generation & Processing

This step builds a graph representation of the road network from the PixelMap and applies postprocessing to simplify and clean the structure.

```bash
docker run --rm \
  -v $(pwd)/DEMO_MATERIAL:/app/input \
  -v $(pwd)/DEMO_MATERIAL/output:/app/output \
  3_graph_generation_and_processing \
  -i /app/input/demo_a1_arese.json
```

**Output directories:**

```
/app/output/GraphMap/
/app/output/GraphMapPostProcessed/
/app/output/Images/GraphMap/
/app/output/Images/GraphMapPostProcessed/
```

These contain the raw and cleaned graph data and their visualizations.

---

### Step 4 — Lanelet2 Conversion

This final step converts the postprocessed graphs into a Lanelet2 HD map.
It requires the Lanelet2 runtime environment (base image `lanelet2:latest`).

```bash
docker run --rm \
  -v $(pwd)/DEMO_MATERIAL:/home/developer/material/demo \
  4_lanelet2_conversion \
  script/graph2let.py -i /home/developer/material/demo/demo_a1_arese.json
```

**Output files:**

```
/home/developer/material/demo/lanenet_output/lanelet2_map.osm
/home/developer/material/demo/lanenet_output/lanelet2_mapinfo.json
```


---

## Notes

* All intermediate and final results will appear under `DEMO_MATERIAL/output/`.
* You can inspect logs by removing `--rm` from the commands to keep the containers after execution or run the images with `-it` option in interactive way and `--entrypoint bash` to overwrite entrypoint and check freely in bash whether any path configuration or execution flow has problem.
>`docker run -it -v ... --entrypoint bash <image_name>`
