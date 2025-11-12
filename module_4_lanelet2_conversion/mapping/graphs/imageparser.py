from typing import Tuple, List
from numpy.typing import NDArray
from .linegraph import LineGraph

class GraphData:
    """Base class for graph data classes returned by Image Parsers
    """
    def __init__(self) -> None:
        self.graphs:List[LineGraph] = []

################################################################################

class ImageParserBase:
    """Base class for all image parsers that generate graphs
    """

    def parse(
        self,
        img: NDArray,
        center_anchor: NDArray,
        rotation_center: Tuple,
        rotation_to_vertical: float = None,
        rotation_to_world: float = None,
        data: GraphData = None
        ) -> GraphData:
        """Parse an image into a graph.
        Args:
            img: image to parse
            center_anchor: position of the center of rotation of the image in 
                world coordinates
            rotation_center: center of rotation of this image in pixel coordinates
                (top left corner is 0,0)
            rotation_to_vertical: rotation in radians to rotate this image to
                try to have as much vertical lines as possibile. Defaults to None.
            data: graph data computed in previous parses. Defaults to None.

        Returns:
            data class containing graphs
        """
        pass