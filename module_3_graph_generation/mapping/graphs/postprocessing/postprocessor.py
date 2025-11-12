from typing import List
from abc import abstractmethod
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import distance


class PostProcessor:
    """Abstract class for graph postprocessing objects
    """

    @abstractmethod
    def process(self, graphs: List[nx.Graph]) -> List[nx.Graph]:
        """Run posprocessing step on the graphs received

        Args:
            graphs: graphs to postprocess

        Returns:
            postprocessed graphs
        """
        pass
    

    def _get_connected_nodes(self, g: nx.Graph, nodes: List[int]) -> List[int]:
        """Get list of nodes that were connected to any of the supplied nodes"""
        out_nodes = []
        for n in nodes:
            out_nodes += [out for out in g[n]]

        out_nodes = list(set(out_nodes)) # delete duplicates
        # filter internal nodels that will be deleted
        out_nodes = [n for n in out_nodes if n not in nodes] 

        return out_nodes

    def _infer_group_class(self, g: nx.Graph, nodes: List[int]) -> int:
        """Find the most common class of a group of nodes
        """
        classes = [self._node_class(g, i) for i in nodes]
        classes, cnts = np.unique(np.array(classes), return_counts=True)

        return classes[np.argmax(cnts)]
    
    def _node_class(self, g: nx.Graph, node: int) -> int:
        """Return node class"""
        return g.nodes[node]["attributes"]["predicted_class"]

    def _get_node_coords(self, g:nx.Graph, nodes: List[int]):
        "Return no coordinates as a (n,2) array"
        coords = [g.nodes[int(c)]["position"] for c in nodes]
        return np.array(coords)
    
    def _get_nearest_node(
        self,
        point: NDArray,
        g: nx.Graph,
        candidates:List[int]
        ) -> int:
        """Return the id of the nearest g node from the candidate list"""
        coords = self._get_node_coords(g, candidates)
        point = np.expand_dims(point, axis=0)
        distances = distance.cdist(point, coords)
        distances = np.squeeze(distances)
        return candidates[np.argmin(distances)]
