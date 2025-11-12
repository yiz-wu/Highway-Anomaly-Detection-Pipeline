
from .postprocessor import PostProcessor
from ...geo import enu_to_latlon
from typing import List
import networkx as nx
from tqdm import tqdm
import numpy as np

import utm

class EnuToUtmProcessor(PostProcessor):
    """Convert graph coordinates from enu to utm"""

    def __init__(self, ref_lat: float, ref_lon:float, ref_h = 0) -> None:
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.ref_h = ref_h
    
    def process(self, graphs: List[nx.Graph]) -> List[nx.Graph]:

        for g in tqdm(graphs):
            for n in list(g.nodes):
                pos = g.nodes[n]["position"]

                lat, lon = enu_to_latlon(pos[0], pos[1], self.ref_lat, self.ref_lon)
                x, y, _, _ = utm.from_latlon(lat, lon)
        
                z = None
                if len(pos) > 2:
                    z = pos[2] + self.ref_h
                
                if z is None:
                    pos = [x,y]
                else:
                    pos = [x,y,z]

                g.nodes[n]["position"] = pos

        
        return graphs