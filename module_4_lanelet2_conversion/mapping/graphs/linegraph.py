from typing import Dict

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sps

class LineGraph:
    """Unidirected efficient graph designed to hold lines and do efficient
    operations
    """
    def __init__(self) -> None:
        # shape: (n, n)
        self.adjacency: sps.coo_matrix = sps.coo_matrix((0,0), dtype=np.uint8)
        # shape: (n, 2) 
        self.centers: NDArray[np.float32] = None
        # shape: (4, 2, n)
        self.boxes: NDArray[np.shape] = None
        # indexed by node id that is adjiacency idx
        # eg: { 1: attributes dict, 2: ... }
        self.attributes: Dict[Dict] = {} 
          
    def add_node(self, center: NDArray, box: NDArray, attributes) -> int:
        """Add a new node to the graph

        Args:
            center: center position of the node. shape: (x, y)
            box: box corners used for overlapping. row shape: (x,y)
            **attributes: extra attribures saved with the node
        Returns:
            node id inside this graph. Node indexes start from 0
        """
        nodeid = self.number_of_nodes()

        new_size = self.adjacency.shape + np.array([1,1])
        self.adjacency.resize(new_size)

        if self.centers is None:
            self.centers = center
        else:
            self.centers = np.vstack((self.centers, center))
        
        box = box[..., np.newaxis]
        if self.boxes is None:
            self.boxes = box
        else:
            self.boxes = np.dstack((self.boxes, box))

        # attr
        if attributes is not None:
            self.attributes[nodeid] = attributes
        return nodeid
    
    def add_edge(self, source: int, dest: int):
        """Add non directed edge between two existing nodes

        Args:
            source: source node id
            dest: dest node id
        """
        self.adjacency.row = np.hstack((self.adjacency.row, [source, dest]))
        self.adjacency.col = np.hstack((self.adjacency.col, [dest, source]))
        self.adjacency.data = np.hstack((self.adjacency.data, [1, 1]))

    def connected_nodes(self, node) -> NDArray:
        """Get node ids of nodes connected to the one provided

        Args:
            node: node id to seach

        Returns:
            array of ids of the connected nodes
        """
        return self.adjacency.col[self.adjacency.row == node]



    def add_graph(self, other: "LineGraph") -> int:
        """Disjoint union of a graph with this one. This operations copy all
        other graph data into the current one. The merged graph nodes will start
        at from the value of num_nodes() before the union

        Args:
            other: other graph to merge
        
        Returns: first id of the nodes added from 'other' graph.
        """
        offset = self.number_of_nodes()

        new_size = offset + other.number_of_nodes()
        self.adjacency.resize((new_size, new_size))

        # merge indexes
        self.adjacency.row = np.hstack(
            (self.adjacency.row, other.adjacency.row + offset)
        )
        self.adjacency.col = np.hstack(
            (self.adjacency.col, other.adjacency.col + offset)
        )
        self.adjacency.data = np.hstack(
            (self.adjacency.data, other.adjacency.data)
        )

        # merge data 
        self.centers = np.vstack((self.centers, other.centers))
        self.boxes = np.dstack((self.boxes, other.boxes))

        for k, v in other.attributes.items():
            self.attributes[k + offset] = v
        
        return offset

    def number_of_nodes(self) -> int:
        return self.adjacency.shape[0]

    def number_of_edges(self) -> int:
        # graph has no direction so the stored elements are duouble of
        # actual edges
        return int(self.adjacency.nnz / 2)

    def node_bbox(self, nodeid) -> NDArray:
        """Get node bounding box of a node in world coordinates

        Args:
            nodeid: node id used to fetch box

        Returns:
           bounding box of the node as a 4x2 (x,y) array containing the box
           vertices
        """
        return self.boxes[..., nodeid]
    
    def node_center(self, nodeid) -> NDArray:
        """Get node center of a node in world coordinates

        Args:
            nodeid: node id used to fetch the center

        Returns:
            (x,y) array containing center coordinates
        """
        # single center
        if len(self.centers.shape) == 1:
            return self.centers

        return self.centers[nodeid, ...]
    
    def node_attributes(self, nodeid) -> Dict:
        """Get node attributes if present

        Args:
            nodeid: node id used to fetch attribute dict

        Returns:
            node attibute dictionary if exist else None
        """
        if nodeid in self.attributes:
            return self.attributes[nodeid]
        else:
            return None
    
    def set_node_center(self, nodeid, center):
        """Set the center for a node

        Args:
            node: target node id
            center: now node center in (x,y) format
        """
        if len(self.centers.shape) == 1:
            self.centers = center
        else:
            self.centers[nodeid, ...] = center

    def set_node_box(self, nodeid, box):
        """Set the box for a node

        Args:
            node: target node id
            box: new box vertices in (4, 2) - (x,y) format
        """
        self.boxes[..., nodeid] = box

    def set_node_attributes(self, nodeid, attributes):
        self.attributes[nodeid] = attributes

    def clear(self):
        """Delete all nodes and edges of this graph
        """
        self.adjacency.resize((0,0))
        self.centers = None
        self.boxes = None
        self.attributes = {}