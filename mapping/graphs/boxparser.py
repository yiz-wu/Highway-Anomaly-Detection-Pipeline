from dataclasses import dataclass
from typing import List, Tuple
import math

import numpy as np
from numpy.typing import NDArray
import cv2
from scipy.spatial import KDTree

from .imageparser import ImageParserBase, GraphData
from .linegraph import LineGraph
from ..graphs import NodeConnectionMode
from .utils import *


@dataclass
class NodeCache:
    """Data class used for internal caching
    """
    graph: LineGraph
    nodeid: int

    def center(self) -> NDArray:
        return self.graph.node_center(self.nodeid)

    def bbox(self) -> NDArray:
        return self.graph.node_bbox(self.nodeid)

################################################################################

class BoxGraphData(GraphData):
    """Data class used to store graph data and ensure that the parser can
    be used in a multithreaded way using different data classes
    """
    def __init__(self) -> None:    
        super(BoxGraphData, self).__init__() 
        self.graphs:List[LineGraph] = []

        # window iteration cache
        self.prev_nodes: List[NodeCache] = None
        self.curr_nodes: List[NodeCache] = []

        # bev itereation cache
        self.cache_tree: KDTree = None
        self.cache_points: List[NodeCache] = []
        self.new_cache_points: List[NodeCache] = []
        
        # debug
        self.debug_boxes = []
        self.debug_centers = []

        self.debug_h_boxes = []
        self.debug_h_centers = []


################################################################################

class BoxImageParser(ImageParserBase):
    """Image to graph parser based on bounding boxes

    This class operates by taking little subregions of the image to parse and
    generates overlapping rotated boxes that are used to connect the centers
    of the boxes based purely on overlaps. This allows to have a really simple
    and generic way to build lines incrementally 

    This class can be a little tricky to use because has two possibile usages 
    that perform different operations:
        * Single region parse
        * Incremental region parse

    The single region parse is performed is no prior data is passed to the parse
    function. This mode allows to parse a region of the map as as if it is disconnected
    from the world.

    The incremental parse on the other hand takes prior graphs and tries to update
    them with new points found in the new iteration.
    """
    def __init__(
        self,
        window_height: int,
        overlap_pixels: int,
        resolution: float,
        max_horizontal_box_size: int = None,
        old_candidate_distance: float = None,
        min_box_area: int = 1,
        node_conn_mode: NodeConnectionMode = NodeConnectionMode.MULTIPLE,
        debug = False,
    ) -> None:
        """
        All parameters are in pixels
        Args:
            window_height: window heigh used to parse the images. 
                A good starting value could be around 0.5 meters in pixels
            overlap_pixels: overlap between two consecutive windows. Boxes should
                overlap a few pixels to be connect, so tune this parameter 
                carefully. This value should be tuned based on your data and must
                be greater than 0 otherwise no box will ever overlap
            resolution: resolution in meters/pixel of the images that will be
                parsed with an instance of this class
            max_horizontal_box_size: maximum horizontal size of a box before
                is split into sub-boxes. If none window_heigh * 2 is used
            old_candidate_distance: max allowed distance of a node from a 
                previous parse iteration from the current point that allows it 
                to be considered a valid cadidate for a node update or connection.
                This value is in metes. If None window_height in meters is used
            min_box_area: minimum box area to filter out junk items. Defaults to 0.
            node_conn_mode: set to false to force a single backward 
                connection for every node
            debug: set to true and keep debug data that can be displayed with
                debug functions
        """

        self.wh = window_height

        assert overlap_pixels > 0, "Overlap pixel should be a positive value"
        self.overlap_pixels = overlap_pixels

        self.resolution = resolution
        
        if max_horizontal_box_size is None:
            self.min_horintal_size = window_height * 2
        else:
            self.min_horintal_size = max_horizontal_box_size
        
        if old_candidate_distance is None:
            self.old_candidate_distance = window_height * resolution
        else:
            self.old_candidate_distance = old_candidate_distance

        self.min_box_area = min_box_area

        self.node_conn_mode = node_conn_mode

        self.debug = debug

    #########################################
    # Nodes and graph managment
    
    def __create_tracked_graph(
        self,
        data: BoxGraphData,
        center: NDArray,
        box: NDArray,
        node_attributes = None
        ) -> int:
        """Create new graph with a node build with provided data

        Args:
            center: _description_
            box: _description_
            graphs: _description_

        Returns:
            created graph instance
        """
        g = LineGraph()
        g.add_node(center=center, box=box, attributes=node_attributes)
        data.graphs.append(g)
        return g

    def __update_graph_references(
        self,
        collection: List[NodeCache],
        match: LineGraph,
        replace: LineGraph,
        offset: int
        ):
        """Replace graph references inside a cache array. 
        This function is usefull to remove invalid graph references after a merge

        Args:
            collection: list to update
            match: graph to remove
            replace: reference of the new graph
            offset: starting node where the matched graph nodes are placed in 
                replace graph
        """
        for i in collection:
            if i.graph == match:
                i.graph = replace
                i.nodeid += offset

    def __merge_with_old_node(
        self,
        data : BoxGraphData,
        center: NDArray,
        box: NDArray,
        candidate_nodes: List[NodeCache]
        ):
        """
        Try to merge a node with old ones that already exist and are part
        """
        for pn in candidate_nodes:                   
            # center is inside another box -> box area near and can be merged 
            # into a single one
            if is_inside_box(center, pn.bbox()):
            #if is_box_overlapping_percent(pn.bbox(), box, 0.5):

                # update center and box positions
                new_center = (pn.center() + center) / 2
                new_box = (pn.bbox() + box) / 2
                pn.graph.set_node_center(pn.nodeid, new_center)
                pn.graph.set_node_box(pn.nodeid, new_box)

                return pn

        return None


    def __connect_graph_node(
        self,
        data : BoxGraphData,
        center: NDArray,
        box: NDArray,
        candidate_nodes: List[NodeCache],
        node_attributes = None
        ) -> NodeCache:
        """Connect a node to the available grapns, if no connection can be made
        a new graph is created
        """
        nodeid = 0
        
        if candidate_nodes is None:
            # no graphs, every box create a new graph
            #print("Created new graph (S)")
            graph = self.__create_tracked_graph(data, center, box, node_attributes)
            return NodeCache(graph, 0)

        # try to connect to old graphs
        base_graph = None
        # TODO: can be vectorialized for speed?
        for pn in candidate_nodes:
            # check if boxes overlap to connect them
            if not is_overlaping2D(box, pn.bbox()):
                continue                      

            # first graph match found, simply add item to graph
            if base_graph is None:
                base_graph = pn.graph 
                nodeid = base_graph.add_node(
                    center=center,
                    box=box,
                    attributes=node_attributes
                )
                base_graph.add_edge(nodeid, pn.nodeid)
            else:
                # we need to check if we need to merge two graphs
                if pn.graph == base_graph:
                    # same graph, just connect nodes
                    base_graph.add_edge(nodeid, pn.nodeid)
                    #print("Multi-point append")
                else:
                    # different graph, need to merge and update caches
                    try:
                        # drop merged graphs
                        old_graph = data.graphs.pop(data.graphs.index(pn.graph))
                    except:
                        print("Triyng to merge a node with an untrakced graph")
                        continue
                    
                    offset = base_graph.add_graph(pn.graph)
                    base_graph.add_edge(nodeid, pn.nodeid + offset)

                    # update caches to delete old graph and re-offeset node ids
                    self.__update_graph_references(
                        data.prev_nodes, old_graph, base_graph, offset
                    )
                    self.__update_graph_references(
                        data.curr_nodes, old_graph, base_graph, offset
                    )
                    self.__update_graph_references(
                        data.cache_points, old_graph, base_graph, offset
                    ) 
                    self.__update_graph_references(
                        data.new_cache_points, old_graph, base_graph, offset
                    )           
            
            # force single backward link of every node
            if self.node_conn_mode == NodeConnectionMode.SINGLE:
                break

        if base_graph is None:
            #print("Created new graph")
            base_graph = self.__create_tracked_graph(data, center, box, node_attributes)
            nodeid = 0

        return NodeCache(base_graph, nodeid)


    #########################################
    # Contour and box operations
    
    def __contour_to_rotated_rects(self, contour) -> Tuple[List[NDArray], bool]:
        """Compure rotated rects from this contour. This function may split
        the contours in little blocks if the contours is horizontal

        Args:
            contour: contour to translate into rotated rects

        Returns:
            list containing rotated rects and a bool that is true if the contour
            was horizontal an thus was split
        """
        # rect format:
        # (center (x,y), (width, height), angle of rotation - clockwise)
        # the rect is taken with x up and y left + clockwise angle
        rect = cv2.minAreaRect(contour)
        hor = False

        rects = [rect]

        # check if box is vertical or horizontal
        # if horizontal we subsplit the box into little boxes
        # to keep the graph smoother
        if abs(rect[1][1]) >= self.min_horintal_size:
            rects = split_horizontal_rect(
                rect,
                self.wh,
                self.overlap_pixels
            )
            hor = True
        
        return rects, hor

    def __rect_to_rotated_bbox(
        self,
        rect: Tuple,
        window_offset: float,
        rot: NDArray,
        rot_center: NDArray
        ):
        """Convert a cv2 rect to 4 points of a rotated bounding box
        
        This function also returns the center of the rotated box.
        """
        box = cv2.boxPoints(rect) # rotated rect around contour
        box += (0, window_offset)
        if rot is not None:
            box = (rot @ (box - rot_center).T).T + rot_center

        # center of every box
        center = box.sum(0) / 4  

        return box, center
    
    def __find_prediction_class(self, area, rect) -> int:
        """Find non zero prediction for the region of area delimited by the given
        rect

        Returns the id of the class that appears most of the time inside the area
        of interest
        """
        mask = np.zeros_like(area)
        points = cv2.boxPoints(rect).astype(np.int32)
        mask = cv2.drawContours(mask, [points], 0, 1, cv2.FILLED)

        cid, count = np.unique(area * mask, return_counts=True)

        # skip 0 class that is background
        # a node should no be the background class
        # may happen that a fallback class is backgroun
        if cid[0] == 0:
            if len(cid) > 1:
                return cid[np.argmax(count[1:]) + 1]
            else:
                return 0
        else:
           return cid[np.argmax(count[1:]) + 1] 


    #########################################
    # cache operations

    def __build_cache(self, data: BoxGraphData):
        """Build cache for current bev iteration
        """
        if data.cache_points == None or len(data.cache_points) <= 0:
            data.cache_tree = None
        
        # build tree with tracked points of previous image
        points = [p.center() for p in data.cache_points]
        
        data.cache_tree = KDTree(points)
    

    def __pick_previous_candicates(
        self,
        data: BoxGraphData,
        center: NDArray,
        max_dist: float,
        k: int = 3
        ) -> List[NodeCache]:
        """Select additional candicates coming from previous bev
        """
        if data.cache_tree is None:
            return None

        _, nodes = data.cache_tree.query(center, k=k, distance_upper_bound=max_dist)

        return [data.cache_points[n] for n in nodes if n != data.cache_tree.n]

    def __update_end_of_iter_cache(
        self,
        data: BoxGraphData,
        track_center: NDArray,
        track_dist: float,
        ):
        """Update cache

        Args:
            data: graph data cache to update
            track_center: position in world coordinates of current iteration
            track_dist: max distance within points are tracked and kept for future
                iterations
        """
        # update tracked points
        # track most of the points of the last image
        if data.cache_tree is not None:
            valid = data.cache_tree.query_ball_point(track_center, r=track_dist)

            # maybe a numpy array is a bit faster
            data.cache_points = [data.cache_points[i] for i in valid]
        else:
            data.cache_points = []
        
        data.cache_points += data.new_cache_points

    #########################################

    def __parse_contours(
        self,
        area,
        contours,
        data: BoxGraphData,
        window_offset: int,
        center_anchor: NDArray,
        rotation_center: NDArray,
        to_world_rot: float,
        fallback_area: NDArray = None
        ):
        """Parse contours into graph nodes

        Args:
            area: image area where contours where computed
            contours: contours to parse
            data: graph data to update
            window_offset: window offset from top of image
            center_anchor: rotation center in world coordinates
            rotation_center: center of rotation in pixels
            to_world_rot: rotation to apply to go back to world rotation. May be
                None if no rotation should be applied
            fallback_area: area used to detect fallback class. Defaults to None
                if no fallback can be computed
        """
        # parse contours into graphs
        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_box_area:
                continue
            
            rects, hor = self.__contour_to_rotated_rects(cnt)

            # parse every box found in the subregion
            internal_iter_cache = []
            for rc in rects:
                box, center = self.__rect_to_rotated_bbox(
                    rc,
                    window_offset,
                    to_world_rot,
                    rotation_center
                )


                if self.debug:
                    if hor:
                        data.debug_h_boxes.append(box)
                        data.debug_centers.append(center)
                    else:    
                        data.debug_boxes.append(box)
                        data.debug_centers.append(center)

                node_class = self.__find_prediction_class(area, rc)
                fallback_class = None
                if fallback_area is not None:
                    fallback_class = self.__find_prediction_class(fallback_area, rc)

                assert node_class != 0, "Somthing went wrong in class detection"\
                    ". A node cannot have background class"

                # translate pixel coodinates to world postions
                assert center.shape == (2, )

                world_box = pixel2worldcoord(
                    box - rotation_center,
                    center_anchor, 
                    self.resolution
                )
                world_center = pixel2worldcoord(
                    center - rotation_center,
                    center_anchor,
                    self.resolution
                )

                # try to merge with old nodes before trying to create a new node
                node_cache = None

                extra_candidates = self.__pick_previous_candicates(
                    data,
                    world_center,
                    max_dist = self.old_candidate_distance
                )

                if extra_candidates is not None:
                    node_cache = self.__merge_with_old_node(
                        data,
                        world_center,
                        world_box,
                        extra_candidates
                    )

                # for horizontal graphs we want to connect also on same 
                # level nodes  not only the one coming from the previous 
                # iteration
                # Note: when building the cadidate list the order of evaluation
                # is really important. High priority nodes should be evaluated
                # first (start of the candidate list)
                candidate_nodes = valid_list_append(
                    internal_iter_cache,
                    data.prev_nodes
                )
                
                # create and update graphs if this node was not merged
                if node_cache is None:

                    candidate_nodes = valid_list_append(
                        extra_candidates,
                        candidate_nodes
                    ) 

                    attrs = {"predicted_class": node_class}  
                    if fallback_class is not None:
                        attrs["fallback_class"] = fallback_class        

                    node_cache = self.__connect_graph_node(
                        data,
                        world_center,
                        world_box,
                        candidate_nodes,
                        attrs
                    )

                    data.new_cache_points.append(node_cache)

                internal_iter_cache.append(node_cache)
                data.curr_nodes.append(node_cache)


        # iteration end for this windows
        # update caches
        data.prev_nodes = data.curr_nodes
        data.curr_nodes = []
        

    def parse(
        self,
        img: NDArray,
        center_anchor: NDArray,
        rotation_center: Tuple,
        rotation_to_vertical: float = None, # radians
        rotation_to_world: float = None,
        data: BoxGraphData = None
        ) -> BoxGraphData:
        """Parse an image into a graph.

        This function can be used in both single and incremental mode

        For best result most of the lines should be vertical. Use the two rotation
        parameters to rotate to a vertical image and then rotate back to the world.
        There are to paramters that allows to have asymmetric rotations. Most 
        of the times rotation_to_world would be -rotation_to_vertical

        Args:
            img: image to parse
            center_anchor: position of the center of rotation of the image in 
                world coordinates
            rotation_center: center of rotation of this image in pixel coordinates
                (top left corner is 0,0 )
            rotation_to_vertical: rotation in radians to rotate this image to
                try to have as much vertical lines as possibile. Defaults to None.
            data: graph data computed in previous parses. Fill this value to 
                parse in incremental mode. Defaults to None (single mode).

        Returns:
            data class containing graphs, caches for incremental mode and a few 
            debug variables filled only if debug=True is set
        """

        if data is None:
            data = BoxGraphData()
        else:
            self.__build_cache(data)

        data.new_cache_points = []

        fallback_img = None

        if len(img.shape) == 3:
            best_class = np.argmax(img, axis=-1)
            # delete max from img to pick second best
            rm = (np.arange(img.shape[-1]) == best_class[...,None]) 
            img[rm] = 0
            fallback_img = np.argmax(img, axis=-1)
            img = best_class

            fallback_img,_ = rotate_image(fallback_img, rotation_to_vertical, rotation_center)
            
        img, _ = rotate_image(img, rotation_to_vertical, rotation_center)
        

        if rotation_to_world is not None:
            to_world_rot = rotation_matrix2D(rotation_to_world)

        img_h = img.shape[0]
        subwindows = int(img_h / (self.wh - self.overlap_pixels))

        #print(f"Using {subwindows} windows for {img_h} pixels")
        for i in range(subwindows):
            # compute window sizing and inspect area
            v_start = i * (self.wh - self.overlap_pixels)
            # fit last window to extra points, not the best but windows height
            # may not be always a factor of area height
            v_end = v_start + self.wh if i + 1 != subwindows else img_h

            v_start = int(v_start)
            v_end = int(v_end)

            subregion = img[v_start:v_end , :] # slice
            fallback_subregion = None
            if fallback_img is not None:
                fallback_subregion = fallback_img[v_start:v_end , :]

            # find contours
            contours, _ = cv2.findContours(
                (subregion != 0).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if len(contours) > 0:
                self.__parse_contours(
                    subregion,
                    contours,
                    data,
                    v_start,
                    center_anchor,
                    rotation_center,
                    to_world_rot,
                    fallback_subregion
                )


        self.__update_end_of_iter_cache(
            data,
            center_anchor,
            np.max(img.shape) * self.resolution
        )

        return data