import numpy as np
from numpy.typing import NDArray
from tqdm import trange
from typing import List, Dict
import networkx as nx
from scipy.spatial import KDTree

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from ... import GraphMap
from ...dataloader import DataLoader

class ElevPointCache:
    """Helper class to make operations on levation"""
    def __init__(self, g: nx.Graph, node: int) -> None:
        self.g: nx.Graph = g
        self.node: int = node

        self.__elevations: List[float] = None
        self.__weights: List[float] = None

    def add_elevation(self, z: float, weight: float):
        if self.__elevations is None:
            self.__elevations = []
            self.__weights = []
        
        self.__elevations.append(z)
        self.__weights.append(weight)
    
    def elevation(self) -> float:
        if self.__elevations is None:
            return None

        return np.average(self.__elevations, weights=self.__weights)

    def attrs(self) -> Dict:
        return self.g.nodes[self.node]
    
    def pos(self) -> NDArray:
        return np.asarray(self.attrs()["position"])


class GpsElevationAnnotator:
    """This class is a simple elevation annoator for every node of the graph.
    Nodes are annotate with the weighted average of near gps positions.

    The weight of every elevetion used to compute the average is computed as
    1 / (distance(node, gps_position) ^ gamma).

    The elevation is added as "ele": <float> attribute for every node that is found
    inside the scan box. Note: if your scan box is too little a few points may 
    have "ele" attribute set to None.
    The "ele" attribute if present is used when converting to lanelet2.

    If you want you can allow this script to correct missing elevation by assigning
    the elevation of the closest node to the node that do not have one.
    """

    def __init__(self,
        dataloader: DataLoader,
        box_size  = (20, 20),
        gamma = 1
        ) -> None:
        """
        Args:
            dataloader: dataloader used to get gps positions
            box_size: (w,h) in meters of the box built around the car gps postions
                to search points. Node: the box h is aligned with the car forward
                direction. Defaults to (20, 20).
        """
        self.dataloader = dataloader
        self.box_size = box_size
        self.gamma = gamma

        self.scan_radius = np.max(box_size)

        self.point_cache: List[ElevPointCache] = []
        self.tree_cache: KDTree = None


    def _build_point_cache(self, graphs: List[nx.Graph]):
        print("Building cache...")
        positions = []
        for g in graphs:
            for n in list(g.nodes):
                pc = ElevPointCache(g, n)
                self.point_cache.append(pc)
                positions.append(pc.pos())
  
        self.point_tree = KDTree(positions)

    def __make_box(self, angle, car_pos,box_size) -> Polygon:
        """Create a box shape around car line this:
            x----------x (h/2, -w/2)
            |     ^    |
            |    car   |
            |          |
            x----------x (-h/2, -w/2)
        
        Args:
            car_dir: direction of the car
            car_pos: car position to place the box in the space
            box_size: (w, h) box sizes in meters
        """

        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])

        w = box_size[0]
        h = box_size[1]

        corners = np.array([
            [-h/2,  w/2], # bottom on car line
            [ h/2,  w/2], # top on car line
            [ h/2, -w/2],
            [-h/2, -w/2]
        ])

        corners = R @ corners.T
        assert corners.shape == (2, 4)

        return Polygon(corners.T  + car_pos)

    def mark_nodes(self, map: GraphMap, autoelevate: bool = False):
        """Add elevation to nodes based on gps z
        Args:
            map: map to annotate
            autoelevate: Set to true to allow this script to correct nodes that 
            do not have an elevation with the elevation of the closest node. 
            Defaults to False.
        """
        graphs = map.nx_graphs()

        self._build_point_cache(graphs)

        # find z and weights for all nodes using scan box
        print("Computing node elevations...")
        for i in trange(len(self.dataloader)):
            gps = self.dataloader.position(i)
            flat_gps = gps[:2]
            elev_gps = gps[-1]
            car_dir = None
            
            # we need to support dataset with bad rotations so we use directions...
            if i > 1:
                car_dir = flat_gps - self.dataloader.position(i-1)[:2]

            # pick point near cars
            candidates = self.point_tree.query_ball_point(flat_gps, self.scan_radius)

            box = None
            # build box around car
            if car_dir is not None:
                angle = np.arctan2(car_dir[1], car_dir[0])
                box = self.__make_box(angle, flat_gps, self.box_size)
            
            # add elevation and weight to valid points
            for c in candidates:
                pt: ElevPointCache = self.point_cache[c]

                # skip if not in box
                if box is not None and not box.contains(Point(pt.pos())):
                    continue
                
                w = 1 / ( np.linalg.norm(pt.pos() - flat_gps) ** self.gamma)
                pt.add_elevation(elev_gps, w)


        # compute weighed elevation and add it as attribute
        for pt in self.point_cache:
            pt.attrs()["ele"] = pt.elevation()
        
        # correct node missing elevation if allowed
        if autoelevate:
            print("Fixing missing elevations...")
            points_to_fix = [pt for pt in self.point_cache if pt.attrs()["ele"] is None]

            while len(points_to_fix) > 0:
                pt = points_to_fix.pop(0)

                print(" " * 30, end="\r")
                print(f"Processing ... {len(points_to_fix)}", end="\r")

                ok = False
                # find close point
                near = self.point_tree.query_ball_point(pt.pos(), self.scan_radius/2, return_sorted=True)
                # check if a point has elevation
                for n in near:
                    nattr = self.point_cache[n].attrs()
                    elev = nattr["ele"] if "ele" in nattr else None
                    if elev is not None:
                        pt.attrs()["ele"] = elev
                        ok = True
                        break
                # no close point has a valid elevation, we need to check later
                if not ok:
                    points_to_fix.append(pt)


        # clear cache
        self.point_cache = []
        self.point_tree = None
