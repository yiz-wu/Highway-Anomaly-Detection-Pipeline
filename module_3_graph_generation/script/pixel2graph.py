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
    dest="input_path",
    default="./scripts/config/monza_11/monza_11_mapping.config.json",
    help="Json preset file path"
)

args = parser.parse_args()
json_path = str(args.input_path)

if not os.path.exists(json_path):
    print(f"Unable to locate file {json_path}")
    exit(1)

with open(json_path, "r") as fp:
    preset = json.load(fp)

print(f"Using preset: {json_path}")

################################################################################
# prepare dataloder for gps positions

from mapping.dataloader import SDFLoader

dt_preset = preset["dataset"]

car_gps_path = dt_preset["car_gps"]
interpolated_path = os.path.join(os.path.dirname(car_gps_path), f"interpolated_{os.path.basename(car_gps_path)}")
if os.path.exists(interpolated_path):
    car_gps_path = interpolated_path

dataloader = SDFLoader(
    dt_preset["images"],
    car_gps_path,
    dt_preset["img_format"] if "img_format" in dt_preset else "png",
    name_regex=dt_preset.get("name_regex", None)
)

start_idx = dt_preset.get("start", None)
if start_idx is not None:
    dataloader.restart_from_(start_idx)

end_idx = dt_preset.get("end", None)
if end_idx is not None:
    dataloader.end_at_(end_idx)

print(f"Found {len(dataloader)} frames")

################################################################################
# Convert pixel map to graph
from mapping import PixelMap, PixelMapIterable
from mapping import GraphMap
import mapping.graphs as mg

gpreset = preset["graph-mapping"]

pixel_map = PixelMap()
pixel_map.load(
    os.path.join(preset["output_root"], "PixelMap")
)


valid_parsers = name2class_map(mg)

min_area_size = gpreset["min_area_size"] if "min_area_size" in gpreset else None

mapitr = PixelMapIterable(
    pixel_map,
    dataloader,
    region_size = gpreset["area_size"],
    ignore_classes=gpreset["ignore_classes"],
    return2d=False,
    min_area_size=min_area_size
    )

parser = valid_parsers[gpreset["parser"]["name"]](**gpreset["parser"]["parameters"]) 

graph_map = GraphMap(parser)

graph_map.from_iterable(
    mapitr,
    gpreset["start"],
    gpreset["end"],
    gpreset["step"]
)

graphs = graph_map.nx_graphs(min_graph_nodes = 2)

graph_map.export(
    os.path.join(preset["output_root"], "GraphMap")
)


################################################################################
# save generated graph map

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

cmap = matplotlib.cm.get_cmap('jet')

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(20, 20, forward=True)
ax.set_aspect('equal', adjustable='box')
fig.suptitle(f"Graph Map")

ax.set_title(f"Extracted {len(graphs)} graphs", pad=25)

for i in range(len(graphs)):
    g = graphs[i]
    coords = nx.get_node_attributes(g, "position")

    color = cmap(float(i)/ len(graphs))
    nx.draw(
        g,
        coords,
        ax,
        node_size=2,
        edge_color=color,
        node_color=color
    ) 


graph_map_output_path = os.path.join(preset["output_root"], "Images/GraphMap")
os.makedirs(graph_map_output_path, exist_ok=True)
map_name = os.path.join(graph_map_output_path, f"graph_map_{gpreset['start']}_{gpreset['end'] if gpreset['end']!=-1 else len(mapitr)}")
plt.savefig(map_name, dpi=300)
print(f"Graph saved at: {map_name}")

# plt.show()


################################################################################
# graph postprocessing

import mapping.graphs.postprocessing as gp

available_processors = name2class_map(gp)

gp_preset = preset["postprocessing"]

if not "stack" in gp_preset:
    print("You must specify a postprocess stack")
    exit(1)

input_path = os.path.join(preset["output_root"], "GraphMap")
post_process_output_path = os.path.join(preset["output_root"], "GraphMapPostProcessed")

processor_stack = []

# generate post-processing stack with specified parameters
for item in gp_preset["stack"]:
    enabled = item["enabled"] if "enabled" in item else True
    if not enabled: # if enalbled = False, skip current iteration (processor)
        continue
    
    processor_name = item["processor"]
    if processor_name not in available_processors.keys():
        print(f"Unknown {processor_name} processor.")
        print(f"Available processors: {list(available_processors.keys())}")
        exit(1)

    params = item["parameters"] if "parameters" in item else None

    proc_class = available_processors[processor_name]
    if params is not None:
        processor_stack.append(proc_class(**params))
    else:
        processor_stack.append(proc_class())

################################################################################
# Post-process

graph_map.postprocess(processor_stack)

min_nodes = gp_preset["min_nodes"] if "min_nodes" in gp_preset else 2
min_nodes = min_nodes if min_nodes > 1 else 2

graphs = graph_map.nx_graphs(min_graph_nodes = min_nodes)

graph_map.export(post_process_output_path, min_graph_nodes=min_nodes)


################################################################################
# save postprocessed graph map


cmap = matplotlib.cm.get_cmap('jet')

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(20, 20, forward=True)
ax.set_aspect('equal', adjustable='box')
fig.suptitle(f"Graph Map")

ax.set_title(f"Extracted {len(graphs)} graphs", pad=25)

for i in range(len(graphs)):
    g = graphs[i]
    coords = nx.get_node_attributes(g, "position")

    color = cmap(float(i)/ len(graphs))
    nx.draw(
        g,
        coords,
        ax,
        node_size=2,
        edge_color=color,
        node_color=color
    ) 


graph_map_output_path = os.path.join(preset["output_root"], "Images/GraphMapPostProcessed")
os.makedirs(graph_map_output_path, exist_ok=True)
map_name = os.path.join(graph_map_output_path, f"graph_map_postprocessed_{gpreset['start']}_{gpreset['end'] if gpreset['end']!=-1 else len(mapitr)}")
plt.savefig(map_name, dpi=300)
print(f"Graph saved at: {map_name}")


