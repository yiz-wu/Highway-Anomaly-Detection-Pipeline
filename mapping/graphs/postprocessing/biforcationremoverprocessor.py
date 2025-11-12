import networkx as nx
import numpy as np
from typing import List
from numpy.typing import NDArray
from .postprocessor import PostProcessor
from tqdm import tqdm

class BiforcationRemoverProcessor(PostProcessor):
    """Postprocessor that removes splits by keeping the longest paths connected
    and disconnecting the other ways.

    This processors is disigned to make complex chains of processors with other 
    classes like HeadConnectorProcessor or DeadEndRemoverProcessor.
    
    This may also be used to prune junk data on simple settings like circuits
    """

    def __init__(self, max_len = None) -> None:
        """
        Args:
            max_len: max allowed len of a segment that can be dropped. 
                Defaults to None to uncostrained len
        """
        super().__init__()
        self.max_len = max_len
    

    def _len_until_biforcation(self, g: nx.Graph, start_node: int):
        """Compute number of nodes (including start node) until a split is found

        You should mask g in a way that start_node has only one connection

        if start_node is also a split 1 is returned
        """
        cnt = 0
        current_node = start_node
        visited  = []

        while current_node is not None:
            visited.append(current_node)
            cnt += 1

            next_nodes = [n for n in g[current_node] if n not in visited]

            if len(next_nodes) == 1:
                current_node = next_nodes[0]
            else:
                current_node = None

        return cnt



    def process(self, graphs: List[nx.Graph]) -> List[nx.Graph]:
        
        result = []

        for g in tqdm(graphs):
            if "type" in g.graph and g.graph["type"] is not None:
                continue

            # pick biforcation nodes only
            eval_pts = [n for n, deg in g.degree() if deg > 2]
            for pt in eval_pts:
                # mask connection to split node
                g_view = nx.restricted_view(g, [pt], [])
                # compute node distances
                connected = np.asarray([n for n in g.neighbors(pt)])
                distances = [self._len_until_biforcation(g_view, n) for n in connected]
             
                # disconnect all chains with shorter lens
                for idx in np.argsort(distances)[:-2]:
                    n = connected[idx]
                    if self.max_len is None or distances[idx] <= self.max_len:
                        g.remove_edge(pt, n)
            
            # separate unconnected regions and save them as separate graphs
            for c in nx.connected_components(g):
                result.append(g.subgraph(c).copy())  

        print(f"Created {len(result) - len(graphs)} new graphs")

        return result
