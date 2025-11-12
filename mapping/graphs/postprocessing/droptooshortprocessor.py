from .postprocessor import PostProcessor

from typing import List
import networkx as nx
from tqdm import tqdm

class DropTooShortProcessor(PostProcessor):
    """Post-processor that drops graphs with not enough nodes. This may be usefull
    on complex post-processing pipelines that require clean up halfway through
    """

    def __init__(self, min_nodes: int) -> None:
        """
        Args:
            min_nodes: minimum number of nodes that a graph must have to be 
                kept
        """
        self.min_nodes = min_nodes
    
    def process(self, graphs: List[nx.Graph]) -> List[nx.Graph]:
        result = []

        for g in tqdm(graphs):
            if len(g) >= self.min_nodes:
                result.append(g)

        print(f"Dropped: {len(graphs)- len(result)} graphs")    

        return result

