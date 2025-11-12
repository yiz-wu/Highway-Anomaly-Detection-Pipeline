from .postprocessor import PostProcessor
from typing import List
import networkx as nx 
import numpy as np
from tqdm import tqdm

class CycleFlattenProcessor(PostProcessor):
    """Class designed to flatten out small node cycles that occur in the map.
    This class replaces loops with points with a average point that keeps thier
    external connectivity
    """

    def __init__(
        self,
        max_cycle_nodes = 6
        ) -> None:
        """
        Args:
            max_cycle_nodes: maximum number of nodes that compose a cycle. 
                Defaults to 6.
        """
        super().__init__()
        self.max_cycle_nodes = max_cycle_nodes
    
    def process(self, graphs: List[nx.Graph]) -> List[nx.Graph]:
        flattened_loops = 0
        dropped_nodes = 0

        for g in tqdm(graphs):
            
            # ignore box graphs generated in other steps
            if "type" in g.graph and g.graph["type"] is not None:
                continue

            cycles = nx.cycle_basis(g)

            while len(cycles) > 0:
                cycle = cycles.pop(0)
                if len(cycle) <= 1 or len(cycle) > self.max_cycle_nodes:
                    continue
                
                # compute center that will replace cycle
                points = np.array([g.nodes[n]["position"] for n in cycle])
                center = np.average(points, axis=0)

                # compute old connectivity
                out_nodes = self._get_connected_nodes(g, cycle)
                new_node_id = max(cycle)

                found_class = self._infer_group_class(g, cycle)

                for n in set(cycle):
                    g.remove_node(n)

                g.add_node(
                    new_node_id,
                    position=center,
                    attributes={"predicted_class": found_class} 
                )

                for n in out_nodes:
                    if n in list(g.nodes):
                        g.add_edge(new_node_id, n)


                # we changes the topography of the graph so we need to update cycles
                cycles = nx.cycle_basis(g)
                
                flattened_loops += 1
                dropped_nodes += len(cycle) - 2


        print(f"Flattened {flattened_loops} cycles - Replaced {dropped_nodes} nodes")

        return graphs
