from .postprocessor import PostProcessor

from typing import List, Union
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

class AutoCircleProcessor(PostProcessor):
    """Postprocessor that tries to search graphs that generate circles
    or semi-circles and replace them with fully resampled circles graphs
    that are marked as shapes.

    This processor works on one graph at a time and ignores near points that 
    belong to different graphs.

    This post-processor uses ransac to fit the circles to data. The generated
    graph have completelly new nodes with the class inferred from the closest
    replaced node.

    Resampled graphs will be marked with "type": "circle" at graph level.

    The resampled nodes will have the same class of the closest node of the old
    graph.

    Inspired to:
    https://github.com/SeongHyunBae/RANSAC-circle-python/blob/master/ransac_circle.py
    """

    # could be better with gradient descent to fit circle
    # https://github.com/sdg002/RANSAC/blob/master/Algorithm/GradientDescentCircleFitting.py

    def __init__(
        self,
        min_sector_angle_deg: float = 180,
        min_nodes: int = 20,
        max_nodes: int = 100,
        max_outliers: int = 5,
        treshold: float = 0.5,
        max_iters: int = 50,
        resample_distance: float = 0.25,
        max_fit_radius: float = 100
        ) -> None:
        """
        Args:
            min_sector_angle_deg: minium angle in degrees of the circular sector
                defined by the nodes of the graph. Defaults to 180 (half circle)
            min_nodes: minium number of nodes of a graph to be considered. 
                Defaults to 10.
            max_nodes: _maximum number of nodes of a graph to be considered.
                Defaults to 100.
            max_outliers: maximum number of points that are not within the 
                threshold. Defaults to 5.
            threshold: max distance in meters around the fitted circle where a 
                point can reside without being cosidered an outlier
            max_iters: number of time the ransac fit is executed. 
                Defaults to 50.
            resample_distance: distance between the generated nodes in meters.
                Defaults to 0.5.
            max_fit_radius: max radius of the circle that can be fit. Defaults to 100
        """
        super().__init__()
        self.min_sector_angle_deg = min_sector_angle_deg
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.max_outliers = max_outliers
        self.treshold = treshold
        self.max_iters = max_iters
        self.resample_distance = resample_distance
        self.max_fit_radius = max_fit_radius

        np.random.seed(1234)

    def _circle_sector_angle(self, points: NDArray, center: NDArray) -> float:
        """Compute the angle of the circle sector defined by the points passed
        as parameter

        Args:
            points: points to use in the computation in (n, 2) shape
            center: center of the circle

        Returns:
            angle of the circle sector defined by points around thier center in
            degrees
        """
        direction = points - center
        angles = np.degrees(np.arctan2(direction[:,1], direction[:,0]))
        #angles[angles < 0] = 360 - angles[angles < 0]

        sector_delta = 360
        # we do not know that is the first point of the sector so we try all of
        # them, sooner or later we will find a minimum angle generated when the
        # lower point of the sector matches the x axis and all other points 
        # follow the 0
        for i in range(angles.shape[0]):
            temp = angles - angles[i]
            temp[temp< 0] = 360 + temp[temp < 0]
            delta = temp.max() - temp.min()

            if delta < sector_delta:
                sector_delta = delta
        #print(sector_delta)
        return sector_delta

    def _random_sample(self, points: NDArray) -> NDArray:
        selected = np.random.randint(points.shape[0], size=3)
        return points[selected, :]

    def _circle_params(self, sample) -> Union[NDArray, float]:
        """Given three points compute center and radius of the circle
        """
        pt1 = sample[0]
        pt2 = sample[1]
        pt3 = sample[2]

        A = np.array([
            [pt2[0] - pt1[0], pt2[1] - pt1[1]],
            [pt3[0] - pt2[0], pt3[1] - pt2[1]]
        ])
        B = np.array([
            [pt2[0]**2 - pt1[0]**2 + pt2[1]**2 - pt1[1]**2],
            [pt3[0]**2 - pt2[0]**2 + pt3[1]**2 - pt2[1]**2]
        ])
        inv_A = np.linalg.inv(A)

        c_x, c_y = np.dot(inv_A, B) / 2
        c_x, c_y = c_x[0], c_y[0]
        r = np.sqrt((c_x - pt1[0])**2 + (c_y - pt1[1])**2)

        return np.array([c_x, c_y]), r

    def _fit_circle(self, points: NDArray) -> Union[NDArray, float]:
        """Find best circle that fits the points and that has a valid number
        of outliers. If nothing is found then None is returned

        Args:
            points: point use to fit a circle

        Returns:
           (center, radius) of the circle if one could be fitted given the
           instance parameters otherwise return None
        """
        best_center = None
        best_radius = None

        best_err = 1e9

        for i in range(self.max_iters):
            try:
                c, r = self._circle_params(self._random_sample(points))
            except:
                # we try to fit random data, inverse will break a lot
                continue

            radiuses = np.linalg.norm(points - c, axis=1)
            
            # fit is valid only if we do not have a lot of outliers
            err_r = np.abs(radiuses - r)
            if np.sum(err_r > self.treshold) > self.max_outliers:
                continue

            # TODO: gradient descend when a good starting fit is found
            err = np.average(err_r)
            if err < best_err:
                best_center = c
                best_radius = r
                best_err = err


        if best_center is None or best_center is None:
            return None

        return best_center, best_radius
    
    def _make_circle_graph(self, c, r) -> nx.Graph:
        """Generate a resampled graph marked as type: circle

        Args:
            c: center of the circle
            r: radius of the circle

        Returns:
            resampled graphs
        """
        g = nx.Graph()

        step = np.arcsin(self.resample_distance / r)
        num_steps = int(np.radians(360) / step)

        last_node = -1

        for i in range(num_steps):
            alpha = step * i
            x = r * np.cos(alpha)
            y = r * np.sin(alpha)

            g.add_node(i,
                position=np.array([x, y]) + c,
                attributes={"predicted_class": 12}
            )

            if last_node != -1:
                g.add_edge(i, last_node)
            
            last_node = i
        
        g.add_edge(last_node, 0)
        
        return g

    def process(self, graphs: List[nx.Graph]) -> List[nx.Graph]:
        
        found_circles = 0

        result = []

        for g in tqdm(graphs):
            sz = g.number_of_nodes()
            if sz < self.min_nodes or sz > self.max_nodes:
                result.append(g)
                continue
            
            # collect points
            points = self._get_node_coords(g, list(g.nodes))

            res = self._fit_circle(points)
            if res is None:
                result.append(g)
                continue
            
            #unpack circle parameters
            center, radius = res

            if radius > self.max_fit_radius:
                result.append(g)
                continue

            if self._circle_sector_angle(points, center) > self.min_sector_angle_deg:
                resampled = self._make_circle_graph(center, radius)
                # move resampled nodes to the graph
                # we updated old graph to not change all references and keep code
                # simple
                resampled.graph["type"] = "circle"

                # use the class of the old nearest node
                for n in resampled.nodes:
                    pos = resampled.nodes[n]["position"]
                    old_nearest = self._get_nearest_node(pos, g, list(g.nodes))
                    cls = g.nodes[old_nearest]["attributes"]["predicted_class"]
                    resampled.nodes[n]["attributes"]["predicted_class"] = cls

                found_circles += 1

                result.append(resampled)
            else:
                result.append(g)


        print(f"Resampled and filled {found_circles} circles")
        
        return result 