from . import ConnectionBaseProcessor
from typing import List
import networkx as nx
from tqdm import tqdm
import numpy as np
import numpy.polynomial.polynomial as poly

class FitReplaceProcessor(ConnectionBaseProcessor):
    """Postprocessor designed to replace agglomate of nodes that form trinagles
    with sampled nodes from a fitted curve that approximate them.
    """
    def __init__(
        self,
        min_nodes:int = 10,
        curve_deg = 3,
        sample_dist = 0.20,
        max_fit_items = None,
        replace_only:List[int] = None,
        min_degree:int = None,
        max_cycle_len:int = 4,
        ) -> None:
        """_summary_

        Args:
            min_nodes: minimum candidates required to perform a fit on them.
                Defaults to 10.
            curve_deg: degree of the curve fitted to the points. Defaults to 3.
            sample_dist: resample distatance of the new points in meters. 
                Defaults to 0.20.
            max_fit_items: Maximum number of candidates to be fit. A group of 
                candidates that exceeds this value is ignored. Defaults to None.
            replace_only: majority class found in a group of candidates that is 
                can be fitted. If the group is not in the required classes it will
                be ignored. Defaults to None.
            min_degree: minimum number of connection that a valid candidate must
                have. Set to None to skip this check
            max_cycle_len: consider only subgrups that form a cycle that has no 
                more that  the specified value. This does not impact the len of 
                the resulting groups. It is useful to prevent long cycles to be 
                considered when generating candidate groups
        """
        super().__init__()
        self.min_nodes= min_nodes
        self.curve_deg = curve_deg
        self.sample_dist = sample_dist
        self.max_fit_items = max_fit_items
        self.replace_only = replace_only
        self.min_degree = min_degree
        self.max_cycle_len = max_cycle_len


    def __fit_and_resample(self, x, y):
        x_order = np.argsort(x)
        x = x[x_order]
        y = y[x_order]


        coeff, diag = poly.polyfit(x, y, self.curve_deg, full=True)

        # resample
        num_samples = int(abs(x[1] - x[-2]) / self.sample_dist) + 1
        new_x = np.linspace(x[1], x[-2], num_samples)
        new_y = poly.polyval(new_x, coeff)

        return new_x, new_y

    def process(self, graphs: List[nx.Graph]) -> List[nx.Graph]:
        
        # https://stackoverflow.com/questions/15721053/whats-the-error-of-numpy-polyfit
        # https://stackoverflow.com/questions/19165259/python-numpy-scipy-curve-fitting
        # https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.fit.html#numpy.polynomial.polynomial.Polynomial.fit

        debug_graphs = []
        replaced_nodes = 0

        for g in tqdm(graphs):
            
            # ignore box graphs generated in other steps
            if "type" in g.graph and g.graph["type"] is not None:
                continue

            groups = self._pick_candidate_groups(
                g,
                min_degree=self.min_degree,
                max_cycle_len=self.max_cycle_len
            )

            for candidates in groups:
                if len(candidates) < self.min_nodes:
                    continue
                
                if self.max_fit_items is not None and len(candidates) > self.max_fit_items:
                    continue

                found_class = self._infer_group_class(g, candidates)
                # check if we can replace the class nodes
                if self.replace_only is not None and found_class not in self.replace_only:
                    continue

                # save old connections
                old_conns = self._get_connected_nodes(g, candidates)

                # fit
                coords = self._get_node_coords(g, candidates)
                
                x = coords[:,0]
                y = coords[:,1]
                delta_x = float(np.max(x) - np.min(x))
                delta_y = float(np.max(y) - np.min(y))

                if delta_x > delta_y:
                    new_x, new_y = self.__fit_and_resample(x, y)
                else:
                    new_y, new_x= self.__fit_and_resample(y, x)

                # replace old nodes with new resampled ones
                start_node = max(g.nodes()) + 1
                g.remove_nodes_from(candidates)

                new_node_ids = []
                for i in range(len(new_x)):
                    node_id = start_node + i
                    g.add_node(
                        node_id,
                        position=np.array([new_x[i], new_y[i]]),
                        attributes={"predicted_class": found_class}
                    )
                    if i > 0:
                        g.add_edge(node_id, node_id -1)
                    
                    new_node_ids.append(node_id)
                

                # connect old items by distance
                for n in old_conns:
                    nearest = self._get_nearest_node(
                        g.nodes[n]["position"],
                        g,
                        new_node_ids
                    )
                    g.add_edge(n, nearest)

                replaced_nodes += len(candidates)

        print(f"Resampled {replaced_nodes} nodes")
        
        return graphs #+ debug_graphs
            