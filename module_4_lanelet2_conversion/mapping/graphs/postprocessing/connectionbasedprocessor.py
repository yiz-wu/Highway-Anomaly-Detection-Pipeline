from .postprocessor import PostProcessor
from typing import List
import networkx as nx 
import numpy as np
from shapely.geometry import LineString
from tqdm import tqdm

class ConnectionBaseProcessor(PostProcessor):
    """Base class for connection based processors that provides a few unility 
    methods to select candidate groups and detect group class
    """

    def _pick_candidate_groups(
        self,
        graph: nx.Graph,
        min_degree:int = 3,
        max_cycle_len:int = None
        ) -> List[List[int]]:
        """Compute a list of grouped candicates that are connected

        Args:
            graph: graph to search connected candidates
            min_degree: minimum number of connection that a valid candidate must
                have. Set to None to skip this check
            max_cycle_len: consider only subgrups that form a cycle that has no 
                more that the specified value. This does not impact the len of 
                the resulting groups. It is useful to prevent long cycles to be 
                considered when generating candidate groups
        """
        cycles = nx.cycle_basis(graph)
        # merge cycles that share at least one node to find grops of nodes that
        # generate a connected cluster in the space
        group_graph = nx.Graph() # use graph to create groups

        for cycle in cycles:
            if len(cycle) <= 1:
                continue
            if max_cycle_len is not None and len(cycle) > max_cycle_len:
                continue

            nx.add_cycle(group_graph, cycle)
        
        if min_degree is not None:
            # delete nodes that only have a connectivity of 2 to prevent
            # to find unwanted loops that mey result in distruptive changes
            # we have to live with a few junk nodes
            degrees = dict(group_graph.degree())
            for node, deg in degrees.items():
                if deg <= min_degree:
                    group_graph.remove_node(node)

        return nx.connected_components(group_graph)