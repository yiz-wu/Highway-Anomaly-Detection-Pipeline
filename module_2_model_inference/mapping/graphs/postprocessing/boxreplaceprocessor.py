from . import ConnectionBaseProcessor
from typing import List
import networkx as nx 
import numpy as np
from shapely.geometry import LineString
from tqdm import tqdm


class BoxReplaceProcessor(ConnectionBaseProcessor):
    """Postprocessor that replaces clusters of point into a rotated box graph 
    that contains all of them.

    Resampled graphs will be marked with "type": box" at graph level

    """
    def __init__(
        self,
        min_cluster_size:int = 5,
        replace_only:List[int] = None,
        min_degree:int = 3,
        max_cycle_len:int = None,
        ) -> None:
        """
        Args:
            min_cluster_size: minimum cluster size to consider. This also becomes
                the minimum number of nodes that a graph need to have to be 
                posprocessed. Defaults to 5.
            replace_only: list opf classes that can be replaced by this processor.
                Defaults to None, this means all detected boxes are replaced
            min_degree: minimum number of connection that a valid candidate must
                have. Set to None to skip this check
            max_cycle_len: consider only subgrups that form a cycle that has no more that 
                the specified value. This does not impact the len of the resulting
                groups. It is useful to prevent long cycles to be considered when
                generating candidate groups
        """
        super().__init__()
        self.min_cluster_size = min_cluster_size
        self.replace_only = replace_only
        self.min_degree = min_degree
        self.max_cycle_len = max_cycle_len
    
    def __graph_box_around(self, coords, prediction_class) -> nx.Graph:
        ls = LineString(coords)
        box = ls.minimum_rotated_rectangle.exterior.coords

        graph = nx.Graph()

        for i in range(4):
            graph.add_node(
                i,
                position= np.array([box[i][0], box[i][1]]),
                attributes= {"predicted_class": prediction_class}
            )
        
        graph.add_edge(0,1)
        graph.add_edge(1,2)
        graph.add_edge(2,3)
        graph.add_edge(3,0)
        graph.graph["type"] = "box"
        
        return graph
    
    
    def __separate_unconnected_subgraphs(self, g: nx.Graph) -> List[nx.Graph]:
        
        components = list(nx.connected_components(g))
        if len(components) == 1: # no need to do anything
            return [g]
        
        subgraphs = [g.subgraph(c).copy() for c in components]
        return [nx.Graph(s) for s in subgraphs]

    def process(self, graphs: List[nx.Graph]) -> List[nx.Graph]:
        
        box_graphs = []
        new_graphs = []
        replaced_node_num = 0
        #debug_candidate_nodes = nx.Graph()
        for g in tqdm(graphs):
            
            if "type" in g.graph and g.graph["type"] is not None:
                new_graphs.append(g)
                continue

            groups = self._pick_candidate_groups(
                g,
                self.min_degree,
                self.max_cycle_len
                )
            
            for candidates in groups:
                if len(candidates) < self.min_cluster_size:
                    continue
                    
                box_class = self._infer_group_class(g, candidates)

                # check if we can replace the class nodes
                if self.replace_only is not None and box_class not in self.replace_only:
                    continue
            
                # prepare data for clustering
                coords = self._get_node_coords(g, candidates)    

                box_graph = self.__graph_box_around(coords, box_class)  
                box_graphs.append(box_graph)

                replaced_node_num += len(candidates)

                g.remove_nodes_from(candidates)
            
            # now we need to check if the current graphs is still valid
            # or has unconnected regions to split
            if g.number_of_nodes() >= 2:
                new_graphs += self.__separate_unconnected_subgraphs(g)

        print(f"Replaced {replaced_node_num} nodes with {len(box_graphs)} boxes")

        return new_graphs + box_graphs #+ [debug_candidate_nodes]

