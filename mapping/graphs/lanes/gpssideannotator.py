
import numpy as np
from tqdm import trange
from typing import List
import networkx as nx
from scipy.spatial import KDTree

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from ... import GraphMap
from ...dataloader import DataLoader


class GPSSideAnnotator:
    """This class is used to annotate graph nodes with their location with respect
    of the car direction.

    The nodes will be annotated with "side": "right/left" based on the direction of
    the car near them

    This class will annotate all the available nodes
    """
    def __init__(self,
        dataloader: DataLoader,
        right_box_size  = (10, 10),
        left_box_size  = (10, 10),
        candidate_radius = 25,
        ) -> None:
        """
        Args:
            dataloader: dataloder providing gps positions to annotate the map
            right_box_size : Size of mark box used to annotate nodes on the righ as 
                ( right, heigh (forward) )in meters.
                Defaults to (10, 10)
            left_box_size : Size of mark box used to annotate nodes on the righ as 
                (left, heigh (forward) )in meters.
                Defaults to (10, 10).
            candidate_radius: radius in meters to use to limit candidate to check
                around every car position
        """
        self.dataloader = dataloader
        self.right_box_size = right_box_size
        self.left_box_size = left_box_size
        self.candidate_radius = candidate_radius

        self.point_data = []
        self.point_tree: KDTree = None


    def _build_point_cache(self, graphs: List[nx.Graph]):

        print("Building cache...")
        positions = []
        for g in graphs:
            for n in list(g.nodes):
                data = g.nodes[n]
                self.point_data.append(data)
                positions.append(data["position"])

        
        self.point_tree = KDTree(positions)

    def __make_box(self, R, car_pos, loc: int, box_size) -> Polygon:
        """If loc == 1 and car direction is up create a box like:
            x----------x
            |          |
            car        |
            |          |
            x----------x
        
        If loc == -1 the box is on the other side of the car
        Args:
            R: car direction rotation matrix
            car_pos: car position to place the box in the space
            loc: 1 or -1 to decide if the box is left or right
        """
        w = box_size[0]
        h = box_size[1]

        corners = np.array([
            [-h/2, 0], # bottom on car line
            [h/2, 0], # top on car line
            [h/2, loc * w],
            [-h/2, loc * w]
        ])

        corners = R @ corners.T
        assert corners.shape == (2, 4)

        return Polygon(corners.T  + car_pos)

    
    def mark_nodes(self, map: GraphMap, debug = False):
        graphs = map.nx_graphs()

        self._build_point_cache(graphs)

        print("Annotating nodes...")
        for i in trange(1, len(self.dataloader)):
            
            # find the car direction

            current_pos = self.dataloader.position(i)
            delta_pos = current_pos - self.dataloader.position(i-1)
            angle = np.arctan2(delta_pos[1], delta_pos[0])

            R = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])

            # limit candidates around car, we do not want to iterate all points
            # in the world
            candidates = self.point_tree.query_ball_point(current_pos[:2], self.candidate_radius)

            point_data = [self.point_data[pt] for pt in candidates]

            # with the direction we create two boxes, one on the right and the other
            # on the left to see if the near points fall inside one of those
            rbox = self.__make_box(R, current_pos[:2], -1, self.right_box_size)
            lbox = self.__make_box(R, current_pos[:2], 1, self.left_box_size)

            for pt in point_data:
                p = Point(pt["position"])
                if rbox.contains(p):
                    pt["side"]= "right"
                    if debug:
                        pt["attributes"]["predicted_class"] = 12
                elif lbox.contains(p):
                    pt["side"]= "left"
                    if debug:
                        pt["attributes"]["predicted_class"] = 0
        
        # clear cache
        self.point_data = []
        self.point_tree = None
