import networkx as nx
import numpy as np
from typing import List
from numpy.typing import NDArray
from .postprocessor import PostProcessor
from tqdm import tqdm
from scipy.spatial import distance, KDTree

class _graphCache():
    def __init__(self, g: nx.Graph, nodes: List[int] = None) -> None:
        self.graph = g

        self.nodes = nodes if nodes is not None else list(g.nodes)

        if len(self.nodes) > 0:
            coords = [g.nodes[int(c)]["position"] for c in self.nodes]
            self.tree = KDTree(np.array(coords))

    def near(self, point: NDArray, max_dist: float) -> List[int]:
        near = self.tree.query_ball_point(
            point,
            max_dist,
            return_sorted=True
            )
        
        return [self.nodes[n] for n in near]
    
    def valid(self) -> bool:
        return len(self.nodes) > 0
        

class HeadConnectorProcessor(PostProcessor):
    """Post-processing that connects node with single connections to a near graph
    """
    def __init__(
        self,
        max_dist: int,
        ignore_shapes:bool = True,
        ignore_classes:bool = None,
        tails_only:bool = False,
        ) -> None:
        """
        Args:
            max_dist:  max distance in meters used to merge graphs
            ignore_shapes: Set to false to also include shape in merge candidates.
                Defaults to True.
            ignore_classes: list of classes to ignore during merge on both graphs.
                Defaults to None.:
            tails_only: set to true to connect only to nodes that have 1 connection
                of the other graphs
        """

        self.max_dist = max_dist
        self.ignore_classes = ignore_classes if ignore_classes is not None else []
        self.ignore_shapes = ignore_shapes
        self.tails_only = tails_only
    
    def _try_merge(
        self,
        main: _graphCache,
        candidates: List[_graphCache],
        heads: List[int]
        ):

        for h in heads:
            point = np.asarray(main.graph.nodes[h]["position"])

            for idx, other in enumerate(candidates):
                
                if not other.valid():
                    continue

                near = other.near(point, self.max_dist)
                # filter by class
                near = [
                    n for n in near 
                        if self._node_class(other.graph, n) not in self.ignore_classes
                    ]

                if len(near) <= 0:
                    continue


                # reindex consistently the other graph
                offset = len(main.graph) 
                other.nodes = [n + offset for n in other.nodes]

                main.graph = nx.disjoint_union(main.graph, other.graph)
                main.graph.add_edge(h, offset + near[0])

                return idx
        
        return None

            


    def process(self, graphs: List[nx.Graph]) -> List[nx.Graph]:
        result = []

        to_process: List[_graphCache] = []

        # pick graphs that should be used to merge and build cache
        for g in graphs:
            # reindex 
            g = nx.convert_node_labels_to_integers(g)

            if self.ignore_shapes and "type" in g.graph and g.graph["type"] is not None:
                result.append(g)
            else:
                if self.tails_only:
                    nodes = [n for n, deg in g.degree() if deg == 1]
                else:
                    nodes = list(g.nodes)
                to_process.append(_graphCache(g, nodes))

        # try merging
        while len(to_process) > 0:
            print(" " * 50, end="\r")
            print(f"Processing... {len(to_process)}", end="\r")
            
            gc = to_process.pop(0)

            if not gc.valid():
                result.append(gc.graph)
                continue

            heads = [
                n for n, deg in gc.graph.degree() 
                    if deg == 1 and 
                    self._node_class(gc.graph, n) not in self.ignore_classes
            ]

            res = self._try_merge(gc, to_process, heads)
            if res is None:
                result.append(gc.graph)
            else:
                to_process.pop(res)

                gc.graph = nx.convert_node_labels_to_integers(gc.graph)

                if self.tails_only:
                    nodes = [n for n, deg in gc.graph.degree() if deg == 1]
                else:
                    nodes = list(gc.graph.nodes)

                to_process.append(_graphCache(gc.graph, nodes))

        print(f"New graph count: {len(result)}")

        return result
