# Convert a pixel map generated with a config into a graph map

import sys
import os
sys.path.append(os.path.abspath("./"))

import json
import argparse
import inspect
import numpy as np

def name2class_map(module):
    members = inspect.getmembers(module, lambda x: inspect.isclass(x) or inspect.isfunction(x))
    return {name: obj for name, obj in members}

################################################################################
# Command args

parser = argparse.ArgumentParser(description='Generate graph map from json preset')

parser.add_argument(
    "-i",
    dest="config_file",
    required=True,
    help="Path to JSON preset configuration file"
)
args = parser.parse_args()

json_path = str(args.config_file)
if not os.path.exists(json_path):
    print(f"Unable to locate file {json_path}")
    exit(1)
with open(json_path, "r") as fp:
    preset = json.load(fp)

print(f"Using preset: {json_path}")

################################################################################
# prepare dataloder for gps positions

from mapping.dataloader import SDFLoader

dataset_cfg = preset.get("dataset")

car_gps_path = dataset_cfg.get("car_gps")
images_folder = dataset_cfg.get("images")
img_format = dataset_cfg.get("img_format")
name_regex = dataset_cfg.get("name_regex")
start_idx = dataset_cfg.get("start", None)
end_idx = dataset_cfg.get("end", None)

interpolated_path = os.path.join(os.path.dirname(car_gps_path), f"interpolated_{os.path.basename(car_gps_path)}")
if os.path.exists(interpolated_path):
    car_gps_path = interpolated_path

dataloader = SDFLoader(
    images_folder,
    car_gps_path,
    img_format,
    name_regex
)

if start_idx is not None:
    dataloader.restart_from_(start_idx)
if end_idx is not None:
    dataloader.end_at_(end_idx)

print(f"Found {len(dataloader)} frames")

################################################################################
# Convert pixel map to graph
from mapping import PixelMap, PixelMapIterable
from mapping import GraphMap
import mapping.graphs as mg
# ---------------------------------------------------------
# Graph Mapping Preset
# ---------------------------------------------------------
gpreset = preset.get("graph-mapping")

output_root = preset.get("output_root", "/app/output")
pixelmap_path = os.path.join(output_root, "PixelMap")

# Load PixelMap
pixel_map = PixelMap()
pixel_map.load(pixelmap_path)

# Load graph mapping parser classes
valid_parsers = name2class_map(mg)

# Extract graph-mapping parameters
area_size        = gpreset.get("area_size")
ignore_classes   = gpreset.get("ignore_classes", [])
start_seq        = gpreset.get("start")
end_seq          = gpreset.get("end")
step_value       = gpreset.get("step")
min_area_size    = gpreset.get("min_area_size")

# Camera/dataloader already defined earlier
mapitr = PixelMapIterable(
    pixel_map,
    dataloader,
    region_size    = area_size,
    ignore_classes = ignore_classes,
    return2d       = False,
    min_area_size  = min_area_size
)

# Parser config
parser_cfg        = gpreset.get("parser")
parser_name       = parser_cfg.get("name")
parser_parameters = parser_cfg.get("parameters")

parser_cls = valid_parsers.get(parser_name)
parser = parser_cls(**parser_parameters)

# Build GraphMap
graph_map = GraphMap(parser)

graph_map.from_iterable(
    mapitr,
    start_seq,
    end_seq,
    step_value
)

# Get graphs
graphs = graph_map.nx_graphs(min_graph_nodes=2)

# Export
graphmap_output_path = os.path.join(output_root, "GraphMap")
graph_map.export(graphmap_output_path)


# ---------------------------------------------------------
# Save generated graph map (Visualization)
# ---------------------------------------------------------
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

cmap = matplotlib.cm.get_cmap('jet')

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(20, 20, forward=True)
ax.set_aspect('equal', adjustable='box')
fig.suptitle("Graph Map")
ax.set_title(f"Extracted {len(graphs)} graphs", pad=25)

for i, g in enumerate(graphs):
    coords = nx.get_node_attributes(g, "position")
    color = cmap(float(i) / len(graphs))
    nx.draw(
        g, coords, ax,
        node_size=2,
        edge_color=color,
        node_color=color
    )

graph_img_path = os.path.join(output_root, "Images/GraphMap")
os.makedirs(graph_img_path, exist_ok=True)

# safer explicit calcs
end_label = end_seq if end_seq != -1 else len(mapitr)
map_name = os.path.join(
    graph_img_path,
    f"graph_map_{start_seq}_{end_label}"
)

plt.savefig(map_name, dpi=300)
print(f"Graph saved at: {map_name}")


# ---------------------------------------------------------
# Post-processing presets
# ---------------------------------------------------------
import mapping.graphs.postprocessing as gp

available_processors = name2class_map(gp)
gp_preset = preset.get("postprocessing")

stack_cfg = gp_preset.get("stack")
if not stack_cfg:
    print("You must specify a postprocess stack")
    exit(1)

processor_stack = []

input_path  = os.path.join(output_root, "GraphMap")
post_output = os.path.join(output_root, "GraphMapPostProcessed")

# Build processor classes
for item in stack_cfg:
    enabled = item.get("enabled", True)
    if not enabled:
        continue

    processor_name = item.get("processor")
    if processor_name not in available_processors:
        print(f"Unknown processor '{processor_name}'")
        print(f"Available processors: {list(available_processors.keys())}")
        exit(1)

    params = item.get("parameters", {})
    proc_class = available_processors[processor_name]

    processor_stack.append(proc_class(**params))


# ---------------------------------------------------------
# Run postprocessing
# ---------------------------------------------------------
graph_map.postprocess(processor_stack)

# extract final parameters
min_nodes = gp_preset.get("min_nodes", 2)
min_nodes = max(min_nodes, 2)

graphs = graph_map.nx_graphs(min_graph_nodes=min_nodes)
graph_map.export(post_output, min_graph_nodes=min_nodes)


# ---------------------------------------------------------
# Save postprocessed graph visualization
# ---------------------------------------------------------
cmap = matplotlib.cm.get_cmap('jet')

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(20, 20, forward=True)
ax.set_aspect('equal', adjustable='box')
fig.suptitle("Graph Map (Postprocessed)")
ax.set_title(f"Extracted {len(graphs)} graphs", pad=25)

for i, g in enumerate(graphs):
    coords = nx.get_node_attributes(g, "position")
    color = cmap(float(i) / len(graphs))
    nx.draw(
        g, coords, ax,
        node_size=2,
        edge_color=color,
        node_color=color
    )

post_img_path = os.path.join(output_root, "Images/GraphMapPostProcessed")
os.makedirs(post_img_path, exist_ok=True)

end_label = end_seq if end_seq != -1 else len(mapitr)
map_name = os.path.join(
    post_img_path,
    f"graph_map_postprocessed_{start_seq}_{end_label}"
)

plt.savefig(map_name, dpi=300)
print(f"Graph saved at: {map_name}")
