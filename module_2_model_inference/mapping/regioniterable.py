from typing import Tuple, List
from numpy.typing import NDArray

class RegionIterable:
    """Base class for region iterators used to generate graph maps starting from
    pixel maps
    """

    def __getitem__(self) -> Tuple[NDArray, NDArray, List[int], float, float]:
        """Returns tuples like:
        (
            img
            center_anchor
            rotation_center
            rotation_to_vertical
            rotation_to_world
        )
        """
        raise NotImplementedError