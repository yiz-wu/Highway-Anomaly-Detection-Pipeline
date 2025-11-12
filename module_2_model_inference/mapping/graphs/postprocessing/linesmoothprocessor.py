from .postprocessor import PostProcessor

from typing import List
import networkx as nx
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray
from scipy.interpolate import splprep, splev

class LineSmoothProcessor(PostProcessor):
    """Smooth lines using n-d spline. This processor tries to keep the control 
    nodes in thier original positions. 

    Control nodes are all the nodes that open a line biforcation or generate
    sharp corners. This nodes are reserved as they are to mantain a high level
    of precision.

    Note: The class of the nodes is mantained as it is and no new nodes are 
    generated.
    """

    @dataclass
    class RootNode:
        id : int
        position: NDArray
        next: int

    def __init__(
        self,
        smooth_coeff: float,
        smooth_shapes: bool = True,
        min_line_nodes: int = 4,
        max_angle_delta_deg: float = 60
        ):
        """
        Args:
            smooth_coeff: Smoothing coefficient
            smooth_shapes: Enable to also smooth shape graphs otherwise skip them.
                Defaults to True.
            min_line_nodes: Minimum nodes that a line require to be smoothed.
                Must be >=4. Defaults to 4.
            max_angle_delta_deg: max allowed delta angle in degrees between two 
                consecutive points of a line. If this angle is exeded new points
                will be cosidered to be part of a different line to smooth. This
                angle is used to preserve sharp corners in lines.
                Defaults to 60.
        """
        self.smooth_coeff = smooth_coeff
        self.smooth_shapes = smooth_shapes
        self.min_line_nodes = min_line_nodes # minimum number of nodes of a line
        self.interrupt_angle = np.radians(max_angle_delta_deg) 

        assert min_line_nodes >= 4, "Minimun number of nodes per line must be >= 4"

    

    def _find_start_node(self, g: nx.Graph) -> int:
        """Find a valid start node that has only 1 connection to another node

        Returns:
            node id inside the graph. Returns None if no candidate is available
        """

        # in shapes any node is a valid starting point
        if "type" in g.graph and g.graph["type"] == "shape":
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

    def _smooth_line(
        self,
        g: nx.Graph,
        rootNode: RootNode = None,
        ):
        """Recursively smooth a graph

        Args:
            g: graph to smooth
            rootNode: root node where the smoothing should start. Defaults to None.

        Returns:
           list of visited nodes
        """
        visited_nodes = []
        selected_nodes = []

        positions = []

        currentNodeId = None

        prev_dir = None

        # find starting node for the current linestring inside the graph
        if rootNode is not None:
            selected_nodes.append(rootNode.id)
            positions.append(rootNode.position)
            currentNodeId = rootNode.next
        else:
            # search a node with only 1 edge
            currentNodeId = self._find_start_node(g)    
        
        while currentNodeId is not None:
            visited_nodes.append(currentNodeId)
            selected_nodes.append(currentNodeId)

            pos = np.asarray(g.nodes[currentNodeId]["position"])
            positions.append(pos)
            
            # find next node; exclude visited nodes to not go backwards in a line
            next_nodes = [n for n in g[currentNodeId] if n not in visited_nodes]
            #TODO: handle loop close?

            # stop iterating if no next node is found
            if len(next_nodes) == 0:
                currentNodeId = None
            # if only one node is found continue normally
            elif len(next_nodes) == 1:
                # check if angle between two adiacent nodes is withing the accepted
                # range otherwise start a new line smooth to preserve the sharp
                # corner
                nxt = next_nodes[0]
                dir = np.asarray(g.nodes[nxt]["position"]) - pos
                # angle is under range -> continue
                if  ( prev_dir is None or 
                      np.abs(self._angle_between(dir, prev_dir))< self.interrupt_angle
                    ):
                    currentNodeId = nxt
                    prev_dir = dir
                else: # -> start new line to fit
                    #print(f"ping: {self._angle_between(dir, prev_dir)}")
                    rn = LineSmoothProcessor.RootNode(currentNodeId, pos, nxt)
                    gw = nx.restricted_view(g, visited_nodes, [])
                    visited_nodes += self._smooth_line(gw, rn)
                    currentNodeId = None
            # biforcation (with 2+ outgoing connections)
            # recursively start other line smmothing executions
            else:
                for n in next_nodes:
                    if n in visited_nodes:
                        continue

                    rn = LineSmoothProcessor.RootNode(currentNodeId, pos, n)
                    gw = nx.restricted_view(g, visited_nodes, [])
                    visited_nodes += self._smooth_line(gw, rn)    
                # if we find a split we need to stop iterating because
                # recursive calls took care of all the following nodes
                break

        # skip short lines
        if len(selected_nodes) <= self.min_line_nodes:
            return visited_nodes
            
        # smooth line excluding control points that are first and last node
        # note: control node do contribute in the spline smoothing but should
        # not updated 
        positions = np.asarray(positions)
        x = positions[:,0]
        y = positions[:,1]
        tck, u = splprep([x, y], s=self.smooth_coeff)
        new_x, new_y = splev(u, tck)

        for i in range(1, len(selected_nodes)-1):
            n = selected_nodes[i]
            new_pos = np.array([new_x[i], new_y[i]])
            g.nodes[n]["position"] = new_pos
            #g.nodes[n]["attributes"]["predicted_class"] = 12

        return visited_nodes


    def process(self, graphs: List[nx.Graph]) -> List[nx.Graph]:
        cnt = 0

        for g in tqdm(graphs):
            gtp = g.graph["type"] if "type" in g.graph else None
            # skip shapes if smooth is disabled
            # skip boxes that do not require smooth
            # and also skip circles that are smooth by construction
            if gtp is not None:
                if not self.smooth_shapes or gtp == "circle" or gtp == "box":
                    continue
            
            self._smooth_line(g)
            cnt += 1
        
        print(f"Smoothed {cnt} graphs")
        return graphs

