# convert graph map to lanelet2 lanestrings

import sys
import os
sys.path.append(os.path.abspath("./"))

import argparse
from dataclasses import dataclass
import networkx as nx
import utm
from typing import List, Dict
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm, trange
import json

import lanelet2 # required to import other subpackages
import lanelet2.core as ll2

from mapping import GraphMap
from mapping.graphs.lanes import LinesSplitPropagator, LaneFinder

parser = argparse.ArgumentParser(description="Convert graph map into Lanelet2 map")

parser.add_argument(
    "-i",
    dest="config_path",
    default="./scripts/config/convert_graph_to_lanelet2.config.json",
    help="Path to JSON configuration file containing all parameters"
)

args = parser.parse_args()
config_path = str(args.config_path)

# ---------------------------------------------------------------------
# 2. Load configuration JSON
# ---------------------------------------------------------------------

if not os.path.exists(config_path):
    print(f"❌ Unable to locate configuration file: {config_path}")
    sys.exit(1)

with open(config_path, "r") as fp:
    config = json.load(fp)

print(f"✅ Using configuration file: {config_path}")

# ---------------------------------------------------------------------
# 3. Extract parameters from the JSON
# ---------------------------------------------------------------------
config_lanelet = config.get("lanelet2")

# Required paths
graph_path = config_lanelet.get("input_path")
output_path = config_lanelet.get("output_file")

if not graph_path or not output_path:
    print("❌ 'input_path' and 'output_file' must be defined in the JSON configuration.")
    sys.exit(1)

# Optional paths / parameters with defaults
annotated_graph_path = config_lanelet.get("annotated_map_path")
zone_number = int(config_lanelet.get("zone_number", 33))
zone_letter = str(config_lanelet.get("zone_letter", "N"))
car_height = float(config_lanelet.get("car_height", 0.0))

# Optional advanced parameters (if you want to move the tuning constants there too)
p_min_lane_size = float(config_lanelet.get("min_lane_size", 1.5))
p_max_lane_size = float(config_lanelet.get("max_lane_size", 6.0))
p_search_tollerance = float(config_lanelet.get("search_tolerance", 0.5))
p_ignore_classes = config_lanelet.get("ignore_classes", [9, 10, 13, 14])
max_corner_angle = np.radians(float(config_lanelet.get("max_corner_angle", 60)))

print("--------------------------------------------------")
print(f"Graph input path:       {graph_path}")
print(f"Output file:            {output_path}")
print(f"Annotated graphs path:  {annotated_graph_path}")
print(f"UTM Zone:               {zone_number}{zone_letter}")
print(f"Car height:             {car_height}")
print(f"Lane size range:        {p_min_lane_size} – {p_max_lane_size}")
print(f"Search tolerance:       {p_search_tollerance}")
print(f"Ignored classes:        {p_ignore_classes}")
print(f"Max corner angle:       {config_lanelet.get('max_corner_angle', 60)}")
print("--------------------------------------------------")

# lanelet generation
lanelet_subtype = "road"
lanelet_location = "urban"

@dataclass
class LineClassMap:
    rtype : str
    rsubtype : str


# define line class mappings to lanelet2 type and subtype
# for existing presets see:
# https://github.com/fzi-forschungszentrum-informatik/Lanelet2/blob/master/lanelet2_core/include/lanelet2_core/Attribute.h
line_classmap = {
    1 : LineClassMap("line_thin", "solid_solid"),
    2 : LineClassMap("line_thin", "solid_dashed"), # exists also the inverse, no dashed_dashed
    3 : LineClassMap("line_thin", "solid"),
    4 : LineClassMap("line_thin", "dashed"),
    9 : LineClassMap("pedestrian_marking", ""), # not sure what changes between pedestrian_marking and zebra_marking
    13: LineClassMap("traffic_sign", "de205"), # precedence 
    14: LineClassMap("stop_line", "")
}

# define conversion settings for shapes of the defined classes. If no class is defined
# the shape is treated as regular line
# At the moment only rectangualr shapes can be converted into a pair of lines
@dataclass
class Shape2LineConversionSettings:
    return_one: bool
    warn: bool = True
    # lanelet can be created only if return_one is set to false
    make_lanelet: bool = True
    let_subtype: str = "road"
    let_location: str = "urban"

shape2line_conversions = {
    9 :  Shape2LineConversionSettings(return_one=False, let_subtype="walkway"),
    14 :  Shape2LineConversionSettings(return_one=True)
}


################################################################################

def set_linestring_attributes(l2_linestring, pred_class, l2_classmap):
    # assign lanelet attributes
    if pred_class in l2_classmap:
        cls = l2_classmap[pred_class]
        l2_linestring.attributes["type"] = cls.rtype
        l2_linestring.attributes["subtype"] = cls.rsubtype
    else:
        print(f"No lanelet2 mapping defined for class :{pred_class}")

def elevated_pos(g: nx.Graph, node: int, default_elev: float):
    """Return node position with elevation if possible otherwise return flat 2d position
    Pass None to default_elev parameter to disable elevation and return flat position
    """
    pos = np.asarray(g.nodes[node]["position"])
    if default_elev is not None:
        elev = default_elev

        if "ele" not in g.nodes[node]:
            print("Warn: Node has no elevation attribute, using origin elevation")
        else:
            e = g.nodes[node]["ele"]
            if e is None:
                print("Warn: Node has null elevation, using origin elevation.")
            else:
                elev = e
    
        pos = np.append(pos, elev)
        
    return pos

def create_l2_pointstring(
    lanefinder: LaneFinder,
    ignore_classes: List[int],
    origin: NDArray,
    l2_classmap: Dict
    ):
    """Create point and linestrings mapping from linestrings that have a class
    that is not ignored.

    Mapping are used to track conversions from internal representation to 
    lanelet2 representation
    """
    point_mappings = {}
    line_mappings = {}

    print("Converting linestrings:")
    for ls in tqdm(lanefinder.linestrings):
        if len(ls) < 2:
            continue

        # use point 1 because the first may be a shared point with another class
        pred_class = ls.attr(1)["attributes"]["predicted_class"]
        if pred_class in ignore_classes:
            continue

        l2_line_points = []
        # create/reference all the points that compose a linestring
        for i in range(len(ls)):
            l2_point = None
            # point already exist, we just need to reference if in the line
            if ls.graph in point_mappings and ls.nodes[i] in point_mappings[ls.graph]:
                l2_point = point_mappings[ls.graph][ls.nodes[i]]
            else:
                # allocate structures
                if ls.graph not in point_mappings:
                    point_mappings[ls.graph] = {}

                # create lanelet point
                default_elev = origin[2] if len(origin) == 3 else None
                position = elevated_pos(ls.graph, ls.nodes[i], default_elev)
                position -= origin                

                l2_point = ll2.Point3d(ll2.getId(), *position)
                # save point mapping
                point_mappings[ls.graph][ls.nodes[i]] = l2_point
            
            l2_line_points.append(l2_point)
        
        # now we have to create a lanelet2 linestring and save its mapping
        l2_linestring = ll2.LineString3d(ll2.getId(), l2_line_points)
        line_mappings[ls] = l2_linestring

        set_linestring_attributes(l2_linestring, pred_class, l2_classmap)

    return point_mappings, line_mappings


def create_l2_lanelets(
    lanefinder: LaneFinder,
    line_mappings,
    lets_subtype,
    lets_location,
    ):
    """Generate lanelets from internal format and return them as array"""
    lanelets = []

    print("Converting lanelets:")
    for lane in tqdm(lanefinder.lanes):
        # get lanelet2 linestrings that compose this lanelet
        # distacrted linestring (too short, should discard lanelets too)
        if lane.left not in line_mappings or lane.right not in line_mappings:
            continue

        r = line_mappings[lane.right]
        l = line_mappings[lane.left]

        lanelet = ll2.Lanelet(ll2.getId(), l, r)
        lanelets.append(lanelet)

        #tag lanelet
        lanelet.attributes["subtype"] = lets_subtype
        lanelet.attributes["location"] = lets_location

    return lanelets


def rect2linestring(
    g : nx.Graph,
    origin: NDArray,
    settings: Shape2LineConversionSettings,
    l2_classmap,
    ) :
    """Convert a rectangular shape to one or a pair of linestrings along the longest
    dimension

    Args:
        g: graph (shape) to translate to linestrings
        origin: origin of the lanelet map. Must have the same dimension of graph
            postion.
        settings: settings used to generate the linestrings
    """

    # compute ordered list of vertices that compose the shape
    vertices = []
    valid_nodes = list(g.nodes)

    current_node = valid_nodes[0]
    cls = g.nodes[current_node]["attributes"]["predicted_class"]

    while current_node != None:
        vertices.append(current_node)
        next_nodes = [n for n in g[current_node] if n not in vertices]
        current_node = next_nodes[0] if len(next_nodes) > 0 else None
    
    # compute edges and respective lenght
    edges = []
    edges_lenght = []
    for i in range(len(vertices)):
        curr = valid_nodes[i]
        nxt = valid_nodes[i + 1 if i + 1 < len(vertices) else 0]
        p0 = np.asarray(g.nodes[curr]["position"])
        p1 = np.asarray(g.nodes[nxt]["position"])
        dist = np.linalg.norm(p1-p0)

        edges.append((curr, nxt))
        edges_lenght.append(dist)

    # find two longest edges
    edges = [edges[i] for i in np.argsort(edges_lenght)[-2:]]

    lines = []
    lanelet = None

    default_elev = origin[2] if len(origin) == 3 else None

    if settings.return_one:

        pos0 = (elevated_pos(g, edges[0][0], default_elev) + elevated_pos(g, edges[1][1], default_elev))/2
        pos0 -= origin

        pos1 = (elevated_pos(g, edges[0][1], default_elev) + elevated_pos(g, edges[1][0], default_elev))/2
        pos1 -= origin

        p0 = ll2.Point3d(ll2.getId(), *pos0)
        p1 = ll2.Point3d(ll2.getId(), *pos1)

        line = ll2.LineString3d(ll2.getId(), [p0,p1])
        set_linestring_attributes(line, cls, l2_classmap)
        lines.append(line)
    else:
        # convert all edges to lines with the same class
        for e in edges:
            pos0 = elevated_pos(g, e[0], default_elev) - origin
            p0 = ll2.Point3d(ll2.getId(), *pos0)

            pos1 = elevated_pos(g, e[1], default_elev) - origin
            p1 = ll2.Point3d(ll2.getId(), *pos1)

            line = ll2.LineString3d(ll2.getId(), [p0,p1])
            set_linestring_attributes(line, cls, l2_classmap)
            lines.append(line)

        if settings.make_lanelet:
            lanelet = ll2.Lanelet(ll2.getId(), lines[0],lines[1])
            # tag lanelet
            lanelet.attributes["subtype"] = settings.let_subtype
            if settings.let_location != "":
                lanelet.attributes["location"] = settings.let_location
    
    return lines, lanelet

################################################################################
# Load map and search lanes

map = GraphMap(None)
map.load(graph_path)

splitter = LinesSplitPropagator(
    min_lane_size=p_min_lane_size,
    max_lane_size=p_max_lane_size,
    tollerance=p_search_tollerance,
    ignore_classes=p_ignore_classes,
    min_node_path_len=10,
)

lanefinder = LaneFinder()
lanefinder.find_lanes(map, splitter)

if annotated_graph_path is not None:
    lanefinder.annotate_linestrings()
    lanefinder.annotate_lanes()
    map.export(annotated_graph_path)


################################################################################
# Convert to lanelet 2 and save to disk

# to project in the space we need to pick an origin for the map
# this point can be anywhere near the map. To properly project the map in the world
# lat/lon coordinate or (utm zone) must be known

# be default we pick the fist node we find because we assume we know the utm zone
# where the point is located
origin_data = list(map.nx_graphs()[0].nodes(data=True))[0][1]
origin = origin_data["position"]
print("origin_data: ", origin_data)
print("origin: ", origin)

# origin_wgs84 = utm.to_latlon(origin[0], origin[1], zone_number, zone_letter)

origin_wgs84 = origin


if "ele" in origin_data and origin_data["ele"] is not None:
    print("Detected map elevation")
    origin = np.append(origin, car_height) # remove car height or lidar wont align
    origin_wgs84 =[*origin_wgs84, car_height]



# now that we picked an origin we can convert our graph map and internal lane
# representation to lanelet 2

pt_map, line_map = create_l2_pointstring(
    lanefinder,
    list(shape2line_conversions.keys()), # ignore classes that must be converted from shapes to lines
    origin, 
    line_classmap
)

lanelets = create_l2_lanelets(lanefinder, line_map, lanelet_subtype, lanelet_location)

# Add data to lenelt map
# TODO: does it check for duplicates or not?
l2_map = ll2.LaneletMap()


for ls in line_map.values():
    l2_map.add(ls)

for l in lanelets:
    l2_map.add(l)


# find skipped shapes and convert shapes to linestrings or lanelets
print("Converting shapes:")
for g in tqdm(map.nx_graphs()):

    if not "type" in g.graph or g.graph["type"] is None:
        continue
    gtype = g.graph["type"]

    if  gtype != "box" and gtype != "shape":
        continue

    n = list(g.nodes)[0]
    cls = g.nodes[n]["attributes"]["predicted_class"]

    if cls not in shape2line_conversions.keys():
        continue
    if g.number_of_nodes() == 4:
        ls, lanelet = rect2linestring(g,origin, shape2line_conversions[cls], line_classmap)
        for l in ls:
            l2_map.add(l)
        if lanelet is not None:
            l2_map.add(lanelet)
    elif shape2line_conversions[cls].warn:
        print(f"WARN: Found a shape that is impossible to translate for class {cls}")


# print generated map info
print("-" * 30)
print(f"Map origin (m): {origin}")
print(f"Map WSG84 (lat, lon, ele): {origin_wgs84}")
print(f"Map points: {len(l2_map.pointLayer)}")
print(f"Map linestrings: {len(l2_map.lineStringLayer)}")
print(f"Map lanelets: {len(l2_map.laneletLayer)}")
print("-" * 30)

################################################################################
# Save lanalet2 map to disk

metadata = { 
    "origin": origin_wgs84,
    "origin_format": "lat lon",
    "zone": f"UTM {zone_number}{zone_letter}"
}

projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(*origin_wgs84))


lanelet2.io.write(output_path, l2_map, projector)

with open(output_path+"info.json", "w") as fp:
    json.dump(metadata, fp)

print(f"Saved lanalet2 map: {output_path}")
