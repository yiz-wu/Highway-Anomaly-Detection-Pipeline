
import sys
import os
sys.path.append(os.path.abspath("./"))

import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import List, Union, Dict, Tuple
from numpy.typing import NDArray
from tqdm import tqdm, trange
from shapely.geometry import Polygon

from mapping import GraphMap
from .splitpropagation import LinesSplitPropagator, PointCache
from .gpssideannotator import GPSSideAnnotator

################################################################################
# helper data classes

@dataclass
class Linestring:
    graph: nx.Graph
    nodes: List[int]
    has_lanelet: bool = False

    def pos(self, idx: int) -> NDArray:
        return np.asarray(self.attr(idx)["position"])
    
    def attr(self, idx: int) -> Dict:
        return self.graph.nodes[self.nodes[idx]]

    def cls(self) -> int: # compute linestring class
        return self.graph.nodes[self.nodes[0]]["attributes"]["predicted_class"]
    
    def contains(self, node: int) -> bool:
        return node in self.nodes

    def is_bounded_by(self, start: int, end: int) -> bool:
        """check if the line has the required start and end nodes even in the 
        opposite order
        """
        fwd = start == self.nodes[0] == start and self.nodes[-1] == end
        bcw = start == self.nodes[-1] == start and self.nodes[0] == end

        return fwd or bcw
    
    def annotate_graph(self, lanelet_id: int):
        for i in range(len(self.nodes)):
            if not "lanelets" in self.attr(i) or self.attr(i)["lanelets"] is None:
                self.attr(i)["lanelets"] = []
            
            self.attr(i)["lanelets"].append(lanelet_id)

    def center_distance(self, other: "Linestring") -> float:
        # check linestring distance is above minimum required
        c_main = (self.pos(0) + self.pos(-1)) / 2
        c_other = (other.pos(0) + other.pos(-1)) / 2

        return np.linalg.norm(c_main - c_other)
    
    def __len__(self):
        return len(self.nodes)

    def __hash__(self) -> int:
        return id(self)


@dataclass()
class Lane:
    right: Linestring
    left: Linestring

    def __eq__(self, other: object) -> bool:
        
        if isinstance(other, Lane):
            same = self.right == other.right and self.left == other.left
            inv = self.right == other.left and self.left == other.right
            return same or inv

        return id(self) == id(other)
    
    def is_ordered(self) -> bool:
        """Return true if right and left nodes can be concatenated directly
        to form a valid shape
        """
        dst_start = np.linalg.norm(self.right.pos(0) - self.left.pos(0))
        dst_end = np.linalg.norm(self.right.pos(0) - self.left.pos(-1))
        # to have an order we have that start-start should form a diagonal 
        # with a higher lenght, If this does not happens then the two lines
        # have the same direction and no not form a valid shape
        return dst_end < dst_start

@dataclass
class LineBiforcationPoint:
    parent_node : int
    start_node: int
    root_graph: nx.Graph

################################################################################

class LaneFinder:
    """Class responsible to find lanes matching different line portions
    """

    def __init__(
        self,
        min_lane_size: float = 2,
        allowed_lane_overlap = 0.3,
        max_hole_len = 6,
        ) -> None:
        """
        Args:
            min_lane_size: minimum size in meters to consider a line valid. 
                Defaults to 2.
            allowed_lane_overlap: max percentual of overlab between two lanes.
                Defaults to 0.3.
            max_hole_len: maximumn allowed nodes that generate a hole in lanes
                that can be fixed automatically. Defaults to 6
        """

        self.linestrings: List[Linestring] = []
        self.lanes: List[Lane] = []

        self.min_lane_size = min_lane_size
        self.allowed_lane_overlap = allowed_lane_overlap
        self.max_hole_len = max_hole_len
        self.hole_segangle_toll = np.radians(5)
    

    def _find_start_node(self, g: nx.Graph) -> int:
        """Find a valid start node that has only 1 connection to another node
        """
        # in shapes any node is a valid starting point
        if "type" in g.graph and g.graph["type"] is not None:
            valid_stats = list(g.nodes())
        else:
            valid_stats  = [nid for nid, deg in g.degree() if deg == 1]

        return valid_stats[0] if len(valid_stats) > 0 else None
    

    def _tracked_linestring(self, g: nx.Graph, nodes: List[int]) -> Linestring:
        ls = Linestring(g, nodes)
        self.linestrings.append(ls)
        return ls

    def _find_linestring(
        self,
        g: nx.Graph,
        parentNode: LineBiforcationPoint = None
        ) -> List[int]:
        """
        Recursively find linestrings and populate class data

        Returns list of visited nodes
        """
        visited_nodes = []
        current_linestring = []

        current_node = None

        root_graph = g

        # find starting node for the current linestring inside the graph
        if parentNode is not None:
            current_linestring.append(parentNode.parent_node)
            current_node = parentNode.start_node
            # always reference top graph, not the views
            root_graph = parentNode.root_graph 
        else:
            # search a node with only 1 edge
            current_node = self._find_start_node(g)
        

        line_class = g.nodes[current_node]["attributes"]["predicted_class"]

        while current_node is not None:
            visited_nodes.append(current_node)
            current_linestring.append(current_node)

            node_data = g.nodes[current_node]
            cls = node_data["attributes"]["predicted_class"]
            is_split = node_data["split"] == "generated" if "split" in node_data else False

            # check for class changes in linestring
            if line_class != cls:
                self._tracked_linestring(root_graph, current_linestring)
                # share the node with next linestring
                current_linestring = [current_node] 

            line_class = cls

            # find next node and connectect linestrings
            # exclude visited nodes to not go backwards in a line
            next_nodes = [n for n in g[current_node] if n not in visited_nodes]

            if len(next_nodes) == 0:
                # stop iterating if no next node is found
                current_node = None
            elif len(next_nodes) == 1:
                # regular line, just check if this node is marked as split
                if is_split:
                    self._tracked_linestring(root_graph, current_linestring)
                    current_linestring = [current_node]
                
                current_node = next_nodes[0]
            else:
                # we reached a biforcation (2+ outgoing links)
                for n in next_nodes:
                    # a loop may have already visited a node
                    if n in visited_nodes:
                        continue

                    sn = LineBiforcationPoint(current_node, n, root_graph)
                    # restrict the graph view allowing to go back to root point
                    # this allow loop closure
                    temp = visited_nodes.copy()
                    temp.pop(temp.index(current_node))

                    hide_links = [(current_node, n)]
                    gw = nx.restricted_view(g, temp, hide_links)

                    visited_nodes += self._find_linestring(gw, sn)
                
                break
        
        # close the linestring at the and or after a biforcation
        self._tracked_linestring(root_graph, current_linestring)

        return visited_nodes
    
    def _build_valid_linestring_index(
        self,
        ignore_classes
        ) -> Tuple[List[Linestring], Dict[nx.Graph, Linestring]]:
        """Create a list of valid class and also return a dict where
        linestrings are grouped by graph for easy and fast indexing
        """
        valid_lines = [ls for ls in self.linestrings if ls.cls() not in ignore_classes]
        line_index = {}
        for ls in valid_lines:
            if ls.graph not in line_index:
                line_index[ls.graph] = []
            
            line_index[ls.graph].append(ls)

        
        return valid_lines, line_index
    
    def _make_lane(self, main: Linestring, other: Linestring):
        """Check if a line match is valid and no duplicate exist. This also reorders
        the linestrings with car annotations is present to align with lanelet2 
        direction requirements
        """
        # we need to check for duplicates because relation annotations are bidirectional
        # so sooner or later we will try to add the same lane two times
        lane = Lane(main, other)
        main.has_lanelet = True
        other.has_lanelet = True
          
        if lane in self.lanes:
            return
        
        if main.center_distance(other) < self.min_lane_size:
            return

        # reorder lines if they have gps annotations and the default order is wrong
        if "side" in other.attr(0) and other.attr(0)["side"] == "right":
            lane.right = other
            lane.left = main
        
        self.lanes.append(lane)

    def _clean_lanes(self):
        """Remove junk lanes using overlaps"""
        boxes: List[Polygon] = []

        def line_vertices(line: Linestring):
            vertices = []
            for i in range(len(line)):
                vertices.append(line.pos(i))
            return vertices

        # create the boxes for overlap checking
        for lane in self.lanes:
            rv = line_vertices(lane.right)
            lv = line_vertices(lane.left)
            p_rl = Polygon(rv + lv)
            p_lr = Polygon(rv + lv[::-1])
            
            if p_rl.is_valid:
                boxes.append(p_rl)
            elif p_lr.is_valid:
                boxes.append(p_lr)
            else:
                boxes.append(None)
        
        # now we can check if we have overlapping lanes. If yes we keep
        # only the smaller ones that are likely to conver only a bit of the
        # road
        valid_lanes_ids = []
        print("Checking generated lanes:")
        for i in trange(len(boxes)):
            if boxes[i] == None:
                continue

            valid = True
            for j in range(0, len(boxes)):
                if boxes[j] == None:
                    continue

                if i == j:
                    continue

                if boxes[i].intersects(boxes[j]):
                    # check only if the current box has a bigger area
                    if boxes[i].area > boxes[j].area:
                        # here we may overlap the other box, if this happens this 
                        # is not a valid lane and should be dropped
                        diff = boxes[i].intersection(boxes[j])
                        # huge overlap
                        if diff.area / boxes[j].area > self.allowed_lane_overlap:
                            valid = False
            if valid:
                valid_lanes_ids.append(i)

        self.lanes = [self.lanes[i] for i in valid_lanes_ids]

        # remark linestrings if they have a lanelet
        for ls in self.linestrings:
            ls.has_lanelet = False

        for lane in self.lanes:
            lane.left.has_lanelet = True
            lane.right.has_lanelet = True     


    def _match_linestrings(self, ignore_classes: List[int]):
        """Match linestring 1-1 to create a lane"""

        print("Building linestring index...")
        lines, lines_index = self._build_valid_linestring_index(ignore_classes)

        broken_attributes_cnt = 0

        for ls in tqdm(lines):
            start = ls.attr(0)
            end = ls.attr(-1)

            if not "split_related" in start or not "split_related" in end:
                broken_attributes_cnt += 1
                continue

            # use stard and end nodes to find if they are both connected to another
            # graph   
            start_rels: List[PointCache] = start["split_related"]
            end_rels: List[PointCache] = end["split_related"]

            for sr in start_rels:
                for er in end_rels:
                    # we found that both our cuts are projectet into another graph
                    # now we just need to find the corresponding linestring on that
                    # graph and we have a lane!
                    if sr.graph == er.graph:
                        if not sr.graph in lines_index:
                            broken_attributes_cnt += 1
                            continue

                        for idl in lines_index[sr.graph]:
                            if idl.is_bounded_by(sr.node_id, er.node_id):
                                # we found the linestring!
                                self._make_lane(ls, idl)
        print(f"Unmatched linestrings: {broken_attributes_cnt} of {len(self.linestrings)}")

        self._clean_lanes()


    def _unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def _angle_between(self, v1, v2):
            """ Returns the angle in radians between vectors 'v1' and 'v2'::
            """
            v1_u = self._unit_vector(v1)
            v2_u = self._unit_vector(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def _fix_lanes_holes(self):
        """Fix small linestrings left behind between lanes by merging them into
        a adiacent linestring that is part of a lanelet
        """
        print("Fixing lane holdes...")
        ls_with_lane: List[Linestring] = []
        ls_to_check: List[Linestring] = []
        ls_index = []
        print("Indexing linestrings...")
        # find linestrings that can be extended and checked in one go
        for idx, ls in enumerate(self.linestrings):
            if len(ls) < 2:
                continue

            if ls.has_lanelet:
                ls_with_lane.append(ls)
            elif len(ls) <= self.max_hole_len:
                ls_to_check.append(ls)
                ls_index.append(idx)

        if len(ls_to_check) == 0:
            print("Nothing to fix")
            return

        to_remove = []    
        
        # merge only with the fist match if any is found
        for i, ls in tqdm(enumerate(ls_to_check)):
            done = False
            for other in ls_with_lane:
                # fast invalid check
                if other.graph != ls.graph or other.cls() != ls.cls():
                    continue
                


                # append on end skipping shared node
                if ls.nodes[0] == other.nodes[-1]:
                    angle = np.abs(
                        self._angle_between(
                            ls.pos(-1) - ls.pos(0),
                            other.pos(-1) - other.pos(0)
                        )
                    )
                    if angle < self.hole_segangle_toll:
                        other.nodes += ls.nodes[1:]
                        to_remove.append(ls_index[i])
                        done = True
                        break
                    
                # append before
                if ls.nodes[-1] == other.nodes[0]:
                    angle = np.abs(
                        self._angle_between(
                            ls.pos(0) - ls.pos(-1),
                            other.pos(0) - other.pos(-1)
                        )
                    )
                    if angle < self.hole_segangle_toll:
                        other.nodes = ls.nodes[:-1] + other.nodes
                        to_remove.append(ls_index[i])
                        done = True
                        break
               
            if done:
                continue
        
        # delete linestrings
        self.linestrings = [ls for i, ls in enumerate(self.linestrings) if i not in to_remove]

        print(f"Fixed {len(to_remove)} holdes")
       

    def find_lanes(
        self,
        map: GraphMap,
        splitter: LinesSplitPropagator,
        gpsannotator: GPSSideAnnotator = None
        ):
        """Generate linestrings and lanes"""

        # propagate cuts and ensure all required information for lane generation
        # is ready
        splitter.mark_nodes(map)
        
        if gpsannotator is not None:
            gpsannotator.mark_nodes(map)
        

        # Generate linestring (skip boxes that cannot generate lanes)
        print("Generating linestrings:")
        for g in tqdm(map.nx_graphs()):
            self._find_linestring(g)

        print(f"Generated {len(self.linestrings)} linestrings")

        # now that we have all the linestring we can match them using split point
        # relations generated by cuts
        self._match_linestrings(splitter.ignore_classes)
        
        # fix small holes by extending linestrings that compose lanelets
        self._fix_lanes_holes()

        print(f"Generated {len(self.lanes)} lanelets")
    

    def generated_lanes(self) -> List[Lane]:
        return self.lanes
    
    def generated_linestrings(self) -> List[Linestring]:
        return self.linestrings


    def annotate_linestrings(self, debug=False):
        """Annotate every node with 'linestrings' attrbute containing a list of
        linestring ids that include the node
        """      
        print("Annotating linestrings...")
        for i in trange(len(self.linestrings)):
            ls = self.linestrings[i]
            for n in ls.nodes:
                if "linestrings" not in ls.graph.nodes[n]:             
                    ls.graph.nodes[n]["linestrings"] = []
                
                ls.graph.nodes[n]["linestrings"].append(i)

                if debug:
                    ls.graph.nodes[n]["attributes"]["predicted_class"] = 12

            
    
    def annotate_lanes(self):
        """Call this function if you want to annotate lanelet ids on graph nodes"""

        print("Annotating lanelets...")
        for i in trange(len(self.lanes)):
            lane = self.lanes[i]
            lane.left.annotate_graph(i)
            lane.right.annotate_graph(i)

        