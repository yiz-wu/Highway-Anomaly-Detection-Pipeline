
# HD Map Generation Pipeline (Dockerized)

This project provides a modular and containerized pipeline for generating HD maps from raw camera images and GPS data. Each step of the process runs inside its own Docker container, so everything is isolated, reproducible, and easy to run on different machines.

Beyond producing standard image-based maps, the pipeline can also exploit custom or pretrained PyTorch segmentation models to detect and highlight highway road anomalies, such as damaged markings or irregular lane structures.

The pipeline takes a folder of images and a GPS log, and step by step produces a final Lanelet2 map (`.osm` + `.json`) describing the detected road markings, lane geometry, and optionally the anomalies identified by the model.


---

## Overview of the pipeline

In a few words, the pipeline does the following:

1. **[Data Preprocessing](docs/modules/1_data_preprocessing.md)** — aligns GPS data with image frames to ensure each image is accurately associated with its spatial position and orientation.  
2. **[Model Inference + BEV Merging](docs/modules/2_model_inference_bev_merging.md)** — runs a segmentation model to detect road markings and anomalies, then projects the results into a bird’s eye view (top-down) pixel map. This representation makes it easier to analyze road geometry and prepare data for graph extraction. 
3. **[Graph Generation + Processing](docs/modules/3_graph_generation_and_processing.md)** — builds a graph representation of the map and post-processes it to clean noise and unstable predictions, replacing dense node clusters with simpler shapes like polygons or splines. This improves map quality and creates smoother features for elements such as crosswalks, STOP markings, and curved lanes. 
4. **[Lanelet2 Conversion](docs/modules/4_lanelet2_conversion.md)** — converts the processed graph into Lanelet2-compatible OSM and JSON files, producing structured map layers ready for future analyze. 

The diagram below shows how data transforms through the stages:

```
GPS + Images
  ↓
1_data_preprocessing                : interpolated pair of data
  ↓
2_model_inference_bev_merging       : Pixel Map (RGB or segmentated output)
  ↓
3_graph_generation_and_processing   : Graph Map (image + json)
  ↓
4_lanelet2_conversion               : Lanelet2 (OSM + JSON)
```

---

## Project structure

```
project_root/
│
├── lanelet2/                        ← build for Lanelet2 base image
├── module_1_preprocessing/          ← docker content for preprocessing
├── module_2_model_inference/        ← docker content for inference + BEV
├── module_3_graph_generation/       ← docker content for graph processing
├── module_4_lanelet2_conversion/    ← docker content for lanelet2 conversion
│
├── mapping/                         ← python package for mapping logic
├── models/                          ← custom segmentation models
│
├── README.md                        ← this file
└── docs/                            ← documentation folder (this section)
    ├── docker_build.md              ← how to build each docker image
    ├── parameters_reference.md      ← detail about configuration parameters
    └── modules/
        ├── 1_data_preprocessing.md
        ├── 2_model_inference_bev_merging.md
        ├── 3_graph_generation_and_processing.md
        └── 4_lanelet2_conversion.md
````

Each module documentation file explains:
- what that part of the pipeline does  
- what it needs as input and what it produces  
- details of its configuration parameters  
- the docker command used to launch it  

---

## Building the images

To build the module images, simply launch

``docker build -t <image_name>:<tag> <path to module folder> ``

To know the details about building process and setting, check the [docker_build.md](docs/docker_build.md).

---

## Running the modules

Each module runs as a separate docker container.
You just need to **mount the input and output folders, and pass the configuration file path inside the container** with `-i`.

An example structure for a folder to be mounted into the container is the following:

```
experiment/
│
├── input/                ← contains all input data
│   ├── images/           ← raw camera frames
│   └── gps_data.csv      ← GPS or vehicle pose information
│
├── output/               ← destination for all generated results
│
└── configuration.json    ← configuration file used by all modules

```
>The proposed folder structure is only a suggestion. You can organize your files differently if you prefer, as long as **the paths defined in your configuration file match your chosen layout, as well as how you gonna mount them to docker module**.



Each docker module (except for last one) has default working directory set at ``/app`` which you could configure in their Dockerfile.

Example run of one module:

```bash
docker run --rm `
  -v /path/to/experiment:/app/experiment `
  1_data_preprocessing `
  -i /app/experiment/configuration.json
```

You can then run the other steps one after another:

```bash
docker run --rm -v ... 1_data_preprocessing -i <path_to_config_file>
docker run --rm -v ... 2_model_inference_bev_merging -i <path_to_config_file>
docker run --rm -v ... 3_graph_generation_and_processing -i <path_to_config_file>
docker run --rm -v ... 4_lanelet2_conversion -i <path_to_config_file>
```

During pipeline execution, several subfolders will automatically be created inside the `output/` directory, for example:

```
output/
├── PixelMap/                  ← pixel-level map generated by the model
├── GraphMap/                  ← graph representation of the map
├── GraphMapPostProcessed/     ← cleaned and optimized version of the graph
├── Lanenet_output/            ← final Lanelet2 OSM and JSON maps
└── Images/                    ← visual representations for each stage
    ├── PixelMap/
    ├── GraphMap/
    └── GraphMapPostProcessed/
```
---

## Configuration file

All modules could use the same JSON configuration file.  
It contains the parameters for every stage, and each module reads only the part it needs (e.g. `dataset`, `bev`, `model`, `lanelet2`, etc.) and ignores the rest.  
You can reuse the same file for entire pipeline.

Example of structure:

```json
{
  "output_root": "/app/output",
  "dataset": {
    "images": "/app/input/images",
    "car_gps": "/app/input/gps_data.csv",
    "img_format": "png",
    "start": 1000,
    "end": 1200
  },
  "bev": { "...": "..." },
  "model": { "...": "..." },
  "pixel-mapping": { "...": "..." },
  "graph-mapping": { "...": "..." },
  "postprocessing": { "...": "..." },
  "lanelet2": { "...": "..." }
}
````

For the detail of parameters you can check [parameters_reference](docs/parameters_reference.md) or corresponding submodule page.

---

