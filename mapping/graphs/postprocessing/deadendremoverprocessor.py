import networkx as nx
import numpy as np
from typing import List
from numpy.typing import NDArray
from .postprocessor import PostProcessor
from tqdm import tqdm

class DeadEndRemoverProcessor(PostProcessor):
    """Post-processing that removes single nodes beforcations that appear due to 
    matching errors in graph junction
    """
    def __init__(self) -> None:
        pass


    def process(self, graphs: List[nx.Graph]) -> List[nx.Graph]:
        rm_count = 0
        for g in tqdm(graphs):

            if "type" in g.graph and g.graph["type"] is not None:
                continue
                
            if len(g) <= 2:
                continue

            # pick biforcations only, we do not care abount other nodes
            candidates = [n for n, deg in g.degree() if deg > 2]

            for c in candidates:
                inspect = g.neighbors(c)
                # pick connected points with one connections
                # this are the nodes we want to remove
                rm = [n for n, deg in g.degree(inspect) if deg == 1]
                for n in rm:
                    rm_count += 1
                    g.remove_node(n)
                    # remove up to cardinality of two then stop
                    if g.degree(c) <= 2:
                        break
        print(f"Removed {rm_count} dead nodes")
        return graphs
