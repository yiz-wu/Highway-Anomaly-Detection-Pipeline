
from ... import GraphMap

import networkx as nx
from scipy.spatial import KDTree
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from typing import List, Union, Dict
from tqdm import tqdm, trange

@dataclass
class PointCache:
    graph: nx.Graph
    node_id: int

    def pos(self) -> NDArray:
        return np.asarray(self.graph.nodes[self.node_id]["position"])
    
    def attr(self) -> Dict:
        return self.graph.nodes[self.node_id]

    def serialize(self):
        return {"graph": self.graph.graph["id"], "node": self.node_id}

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, PointCache):
            return self.graph.graph["id"] == other.graph.graph["id"] and self.node_id == other.node_id
        return id(self) == id(other)

@dataclass
class PointOfInterest:
    point: PointCache
    road_direction : NDArray
    sharp_corner: bool = False

class LinesSplitPropagator:
    """Lanelets required to be atomic chunks of roads with the same rules. This
    requires to propagate splits to near lines to allow them to be split into 
    little linestrings that form a (almost) perpendicular line in the road when
    any of the lines that compose the road has a class change, or a biforcation
    or ends.
    
    This class mark nodes of graphs that could be used to compose lanelets with 
    perpendicular cuts with respect of the road direction. The algorithm searches
    candidates points by using perpendicular scan line in point of
    interest like a class change.

    The candicates node will me marked with `"split": true`
    """
    def __init__(
        self, 
        min_lane_size: float,
        max_lane_size: float,
        tollerance: float = 0.3,
        max_corner_angle_deg: float = 60,
        ignore_classes: List[int] = [],
        min_node_path_len = 5,
        sharp_angle_generate_splits = False,
        robust_direction_stack_size = 3,
        max_propagation_angle_deg: float = 70
        ) -> None:
        """
        Args:
            min_lane_size: Minimum distance in meter of a valid point that can 
                be used as possibile candidate for a lanelet. Defaults to 1.
            max_lane_size: max lenght of scan line used to search canidates.
                This parameter should be the maximum expected road size plus a little
                margin to cover noise. Defaults to 3.
            tollerance: max distance in meters of a node from the scan line 
                to be considered a valid candidate. Defaults to 0.3.
            max_corner_angle_deg: max angle of a corner to be considered sharp
            ignore_classes: list of classes to ignore while propagating splits
            min_node_path_len: minimum path len between two nodes of the same graph
                to be cosidere both valid candidates for split propagation
            sharp_angle_generate_splits: set to true to allow sharp angles to
                generate splits. Defaults to false
            robust_direction_stack_size: number of past direction to keep to
                compute split source point direction. This settings does not apply
                to start points. Defaults to 2
            max_propagation_angle_deg: max angle in degrees to allow between the 
                scan segment and the matched line
        """
        self.min_lane_size = min_lane_size
        self.max_lane_size = max_lane_size
        self.tollerance = tollerance
        self.max_corner_angle = np.radians(max_corner_angle_deg)
        self.ignore_classes = ignore_classes

        self.point_cache : List[PointCache] = []
        self.point_tree : KDTree = None

        self.min_node_path_len = min_node_path_len

        self.sharp_angle_generate_splits = sharp_angle_generate_splits

        self.dir_stack_len = robust_direction_stack_size

        self.max_propagation_angle = np.radians(max_propagation_angle_deg)
    
    ############################################################################
    # Candidate selection

    def _build_point_cache(self, graphs: List[nx.Graph]):
        """Build point cache and corresponding kdtree for fast spatial search
        Point cache index == kdtree index
        """
        positions = []

        print("Building cache...")
        for g in graphs:
            for n in list(g.nodes):
                if g.nodes[n]["attributes"]["predicted_class"] in self.ignore_classes:
                    continue

                pt = PointCache(g, n)
                self.point_cache.append(pt)
                positions.append(pt.pos())
        
        self.point_tree = KDTree(positions)


    def _reindex_and_clean_graphs(self, graphs: List[nx.Graph]):
        """Delete cut attributes and set a graph attribute with a numeric id"""

        print("Indexing graphs...")
        for i in range(len(graphs)):
            graphs[i].graph["id"] = i
            for n in graphs[i].nodes(True):
                if "split" in n:
                    del n["split"]
                if "split_related" in n:
                    del n["split_related"]
                if "lanelets" in n:
                    del n["lanelets"]


    def _get_near_points(self, point: NDArray) -> List[PointCache]:
        """Find a list of PointCache elements inside the circle defined by 
        max_lane_size and min_lane_size.

        This function has the role of a fast discrimination to find only candidates
        that is worth investigate further.

        If nothing if found an empty list is returned
        """
        ids = self.point_tree.query_ball_point(point, self.max_lane_size)

        return [self.point_cache[i] for i in ids if np.linalg.norm(self.point_cache[i].pos() - point) >= self.min_lane_size]


    def _get_scan_segment(self, point :NDArray, road_dir: NDArray) -> List[NDArray]:
        """Compute the two segment edges that define a perpendicular scanline
        passing in the required point
        """
        angle = np.arctan2(road_dir[1], road_dir[0])
        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])

        # since we are intrested in the perpendicular scan line we can pick
        # and rotate vector with the distance on y axis that will be rotated on the
        # direction we are intrested in
        dx = R @ [0, self.max_lane_size] + point
        sx = R @ [0, -self.max_lane_size] + point

        return [dx, sx]
    
    def _point_distance(self, point: NDArray, segment_edges: NDArray) -> float:
        """Compute the distance between a point and a segment defined by its edges
        Based on: https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
        """
        pe = point - segment_edges[0]
        segment = segment_edges[1] - segment_edges[0]

        param = pe @ segment / (np.linalg.norm(segment)**2)

        p = None
        if param < 0:
            p = segment_edges[0]
        elif param > 1:
            p = segment_edges[1]
        else:
            p = segment_edges[0] + param * segment
        
        return np.linalg.norm(point - p)

    def _merge_candidates(self, candidates: List[PointCache]):
        """Use the neighborhood a each candidate to eventually replace it with.
        Swap is made only if the two nodes have the same class. Otherwise the
        candidate is valid
        """
        res = []

        for c in candidates:
            replaced = False
            #cls = c.attr()["attributes"]["predicted_class"]
            for near in nx.neighbors(c.graph, c.node_id):
                near_attr = c.graph.nodes[near]
                #near_cls = near_attr["attributes"]["predicted_class"]

                if "split" in near_attr:
                    replaced = True
                    res.append(PointCache(c.graph, near))
                    break
            
            if not replaced:
                res.append(c)
        return res
    

    def _filter_compatible_candidates(self, candidates: List[PointCache], road_dir: NDArray):
        """Filter out candidates that have a direction that is not similar to 
        the road one
        """
        # a candidate is valid if a rect passing by itself and one of the connect
        # nodes has a similar if rect angular cofficient is close with the one
        # of generated by the direction of the road

        # this check works... its a bit too restrictive
        # this eliminates decent custs that are usefull due to noise

        valid = []
        for c in candidates:
            angles = []
            # compute angle between directions and pick the minimum
            for neighbour in c.graph[c.node_id]:
                pos = c.graph.nodes[neighbour]["position"]
                alpha = np.abs(self._angle_between(c.pos() - pos, road_dir))
                beta = np.abs(self._angle_between(pos - c.pos(), road_dir))
                angles.append(min(alpha, beta))

            if np.any(angles <= self.max_propagation_angle):
                valid.append(c)

        return valid

    def _find_candidates(self, point: NDArray, road_dir: NDArray) -> List[PointCache]:
        """Find candidates near scanline and return them
        """
        near_pt = self._get_near_points(point)
        # nothing to check
        if len(near_pt) <= 0:
            return near_pt

        scan_segment = self._get_scan_segment(point, road_dir)
        dists = [self._point_distance(p.pos(), scan_segment) for p in near_pt]
        
        # sort by distance from scan segment and delete invalid points that are 
        # too far
        dists = np.asarray(dists)
        valid = dists <= self.tollerance

        # weight by distance to scan line center
        weight = [np.linalg.norm(p.pos() - point) for p in near_pt]
        dists *= weight

        order = np.argsort(dists[valid])
        near_pt = np.asarray(near_pt)[valid][order]

        candidates = near_pt.tolist()
        candidates = self._filter_compatible_candidates(candidates, road_dir)
        candidates = self._merge_candidates(candidates)

        return candidates
    
    
    ############################################################################
    # Graph navigation

    def _find_start_node(self, g: nx.Graph) -> int:
        """Find a valid start node that has only 1 connection to another node
        """

        # in shapes any node is a valid starting point
        if "type" in g.graph and g.graph["type"] is not None:
            valid_stats = list(g.nodes())
        else:
            valid_stats  = [nid for nid, deg in g.degree() if deg == 1]

        return valid_stats[0] if len(valid_stats) > 0 else None

    def _unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def _angle_between(self, v1, v2):
            """ Returns the angle in radians between vectors 'v1' and 'v2'::
            """
            v1_u = self._unit_vector(v1)
            v2_u = self._unit_vector(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def _make_poi(
        self,
        graph: nx.Graph,
        node: int,
        dir_stack: List[NDArray],
        is_sharp:bool = False
        ):
        """Helper function used to create a point of intereset for a cut
        """
        #  compute direction as the average of the passed direction
        if len(dir_stack) > 1:
            dir = self._unit_vector(np.sum(dir_stack, axis=0))
            assert dir.shape == (2,)
        else:
            dir = dir_stack[0]

        pc = PointCache(graph, node)
        pt = PointOfInterest(pc, dir, is_sharp)

        return pt
    
    def _is_sharp_corner(self, dir, prev_dir):
        """Check if two direction make a sharp corner"""

        if dir is None or prev_dir is None:
            return False

        return np.abs(self._angle_between(dir, prev_dir)) > self.max_corner_angle

    
    def _shared_biforcation_split_dir(self, g: nx.Graph, current: int) -> NDArray:
        """Compute direction that is perpendicular (or closer) to most nodes. If 
        none can be found then None is returned
        """
        tollerance= 0.05

        directions = []
        current_pos = np.asarray(g.nodes[current]["position"])
        for n in nx.neighbors(g, current):
            d= self._unit_vector(np.asarray(g.nodes[n]["position"]) - current_pos)
            directions.append(d)

        consensus = [0 for _ in directions]
        for i, d1 in enumerate(directions):
            for d2 in directions:
                c = d1 @ d2.T
                if c > 1 - tollerance or c < -1 + tollerance:
                    consensus[i] +=1
        
        consensus = np.asarray(consensus)
        m_cons = np.max(consensus)
        # cannot find a direction with a majority of segments laying on it
        if np.sum(consensus == m_cons) != 1:
            return None
        else:
            return directions[np.argmax(consensus)]

    def _search_point_of_interest(
        self,
        g: nx.Graph,
        startNode: int = None,
        start_dir = None,
        loop_parent = None,
        ) -> Union[List[PointOfInterest], List[int]]:
        """Find point of interest from raw graphs"""
        visited_nodes = []
        poi = []
        currentNodeId = None

        prev_dir = [start_dir] if start_dir is not None else []

        is_shape_graph = (g.graph["type"] if "type" in g.graph else None) is not None

        # find starting node for the current linestring inside the graph
        if startNode is not None:
            currentNodeId = startNode
        else:
            # search a node with only 1 edge
            currentNodeId = self._find_start_node(g)

            # add start point for non shaped graphs (only 1 connection)
            if not is_shape_graph:
                nxt = [n for n in g[currentNodeId] if n not in visited_nodes][0]
                pnext = np.asarray(g.nodes[nxt]["position"])
                pstart= np.asarray(g.nodes[currentNodeId]["position"])
                dir =  self._unit_vector(pnext - pstart)
                poi.append(self._make_poi(g, currentNodeId, [dir]))
                
        
        lineClass = g.nodes[currentNodeId]["attributes"]["predicted_class"]
        
        while currentNodeId is not None:
            visited_nodes.append(currentNodeId)

            position = np.asarray(g.nodes[currentNodeId]["position"])
            cls = g.nodes[currentNodeId]["attributes"]["predicted_class"]

            # check for class changes in linestring
            # if class changes then 
            if lineClass != cls and cls not in self.ignore_classes:
                poi.append(self._make_poi(g, currentNodeId, prev_dir))
            
            lineClass = cls
            
            # find next node and connectect linestrings
            # exclude visited nodes to not go backwards in a line
            next_nodes = [n for n in g[currentNodeId] if n not in visited_nodes]
            # stop iterating if no next node is found
            if len(next_nodes) == 0:
                # add last node only for non shaped graphs
                if lineClass not in self.ignore_classes and not is_shape_graph:
                    poi.append(self._make_poi(g, currentNodeId, prev_dir))
                currentNodeId = None
            # if only one node is found continue normally
            elif len(next_nodes) == 1:
                nxt = next_nodes[0]
                dir = np.asarray(g.nodes[nxt]["position"]) - position
                dir = self._unit_vector(dir)

                # skip closure node of internal loops
                if loop_parent is not None and nxt == loop_parent:
                    currentNodeId = None
                    break

                if (
                        len(prev_dir) > 0 and
                        self._is_sharp_corner(dir, prev_dir[-1]) and
                        lineClass not in self.ignore_classes and
                        self.sharp_angle_generate_splits
                    ):
                        poi.append(self._make_poi(g, currentNodeId, prev_dir, True))

                currentNodeId = nxt
                
                # rememebr only required number of past directions
                prev_dir.append(dir)
                if len(prev_dir) > self.dir_stack_len:
                    prev_dir.pop(0)

            # biforcation (with 2+ outgoing connections)
            else:
                # split point are point of interest
                if lineClass not in self.ignore_classes and not is_shape_graph:
                    # compute direction that has more consensus between nodes. If
                    # none, keep the current one
                    pref_dir = self._shared_biforcation_split_dir(g, currentNodeId)
                    if pref_dir is None:
                        poi.append(self._make_poi(g, currentNodeId, prev_dir))
                    else:
                        poi.append(self._make_poi(g, currentNodeId, [pref_dir])) 

                # recursively navigate sub-lines
                for n in next_nodes:
                    if n in visited_nodes:
                        continue
                    
                    # compute direction from splitting node to next-node after
                    # split
                    dir = np.asarray(g.nodes[n]["position"]) - position
                    dir = self._unit_vector(dir)

                    temp = visited_nodes.copy()
                    temp.pop(temp.index(currentNodeId))

                    gw = nx.restricted_view(g, temp, [(currentNodeId, n)])
                    epoi, vn = self._search_point_of_interest(gw, n, dir, currentNodeId)
                    poi += epoi
                    visited_nodes += vn 

                currentNodeId = None
        
        return poi, visited_nodes
    

    def _search_and_mark_all_poi(self, graphs: List[nx.Graph]) -> List[PointCache]:
        """Search all point of interest that may generate a split and mark them with
        attribute `split`: `source`
        """
        point_of_interest: List[PointOfInterest] = []

        print("Searching split points:")
        for g in tqdm(graphs):
            
            gtype = g.graph["type"] if "type" in g.graph else None
            if gtype is not None and gtype == "box":
                continue

            poi, _ = self._search_point_of_interest(g)
            point_of_interest += poi

        # mark all points with attribute
        for pt in point_of_interest:
            pt.point.attr()["split"] = "source"

        return point_of_interest
        
    ############################################################################

    def _annotate_relation(self, attr_dict, candidate: PointCache):
        if "split_related" not in attr_dict:
            attr_dict["split_related"] = []
        
        relations = attr_dict["split_related"]
        if candidate not in relations:
            relations.append(candidate)

    def mark_nodes(self, map: GraphMap, debug=False):
        """Mark split nodes with attribute `"split": "generated".

        Set debug to true to set point of interest that may generate split to Debug
        class and point where split is propagated with class background.
        This allows to save graphs and see what points were found with split 
        propagation.
        """

        graphs = map.nx_graphs()

        self._build_point_cache(graphs)
        self._reindex_and_clean_graphs(graphs)
        
        point_of_interest: List[PointOfInterest] = []
        point_of_interest = self._search_and_mark_all_poi(graphs)
        print(f"Found {len(point_of_interest)} split points")   

        split_propagation_count = 0
        missing_dir_count = 0

        # disallow to use generator points as new point of interest
        visited_points = [p.point for p in point_of_interest]

        # mark nodes for debug
        if debug:
            for pt in point_of_interest:
                pt.point.attr()["attributes"]["predicted_class"] = 12

        # a possible aternative implementation that may create a lot more points
        # it to use a scan line only in one direction and the recursively propagate
        # the split by adding point to the point_of_interest_list
        print("Propagating splits...")

        #for pt in tqdm(point_of_interest): # len(point_of_interest) > 0:
        while len(point_of_interest) > 0:
            pt = point_of_interest.pop(0)
            visited_points.append(pt.point)

            print(" " * 30, end="\r")
            print(f"Processing ... {len(point_of_interest)}", end="\r")

            graph_points = {}


            if pt.road_direction is not None:
                candidates = self._find_candidates(pt.point.pos(), pt.road_direction)

                # pick only only one directly connected point per graph
                for c in candidates:
                    is_new_point = False
                    skip = False

                    # not poit for graph, just add
                    if c.graph not in graph_points:
                        # if they are on the same graph then we must check path len
                        if pt.point.graph == c.graph:
                            path_len = nx.shortest_path_length(c.graph, c.node_id, pt.point.node_id)
                            if path_len < self.min_lane_size:
                                is_new_point = True
                        else:
                            is_new_point = True
                    else:
                        # check if the new point is connected to a known point
                        # if yes skip the point otherwise is safe to assume that
                        # the lines we are tring to cut merge at some point but
                        # at the cut level they are not connected so the cut
                        # should be propated to both lines
                        for n in graph_points[c.graph]:
                            path_len = nx.shortest_path_length(c.graph, c.node_id, n.node_id)
                            if path_len < self.min_node_path_len:
                                skip = True
                                break     

                    # TODO: elimina candidati sullo stesso grafo e anche vicini al nodo
                    # di partenza      

                    if is_new_point or not skip:
                        # update graph dict to check for near nodes                                      
                        if not c.graph in graph_points:
                            graph_points[c.graph] = []
                        graph_points[c.graph].append(c)
                        
                        # annotate both nodes with relation that we have found
                        self._annotate_relation(pt.point.attr(), c)
                        self._annotate_relation(c.attr(), pt.point)


                        # recursive propagation only for items that were not evaluated
                        if c not in visited_points:
                            point_of_interest.append(PointOfInterest(c, pt.road_direction))
                            split_propagation_count += 1

                            # annotate that this point is a generated split
                            c.attr()["split"] = "generated"

                            if debug:
                                c.attr()["attributes"]["predicted_class"] = 0

            else:
                missing_dir_count += 1

        print(f"Splits propagated to: {split_propagation_count} nodes")
        print(f"Skipped nodes due to missing directions: {missing_dir_count}")

        # clear cache
        self.point_cache = []
        self.point_tree = None

         