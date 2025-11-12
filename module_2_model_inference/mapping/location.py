import numpy as np
from numpy.typing import NDArray
import enum
from typing import Dict
from .dataloader import DataLoader

class OffsetDirection(enum.Enum):
    """Enum describing possible offsets applied to the anchors of the extents
    """
    LEFT_TOP = "left_top"
    """Apply offset to top-left anchor
    """
    RIGHT_BOTTOM = "right_bottom"
    """Apply offset to bottom-right anchor
    """


class MapExtents:
    """Location and extents of a region in the map

    This class can be used in 2 ways.
        The 1st is to instantiate only with resolutions and add points with put_point(). 
        This allow to calculate the region EXTENT that contains those points.

        The 2nd use is as a data holder by manually setting every value 
        of the constructor with manually computed data.

    In either case the resulting istance can be later manipulated.
    """
    def __init__(
        self,
        resolution: float,
        top_left_anchor:NDArray[np.float32] = None,
        bottom_right_anchor:NDArray[np.float32] = None,
        extents:NDArray[np.float32] = None,
        extents_pixel:NDArray[np.int32] = None
        ) -> None:
        """Constructor

        Args:
            resolution: resolution of the map
            top_left_anchor: precomputed top-left anchor. Defaults to None.
            bottom_right_anchor: precomputed bottom-right anchor..Defaults to None.
            extents: precomputed extents in meters. Defaults to None.
            extents_pixel: precomputed extents in pixels. Defaults to None.
        """
        self.resolution = resolution

        # internal anchors are not offset directly
        self.top_left_anchor = top_left_anchor          # min x, max y
        self.bottom_right_anchor = bottom_right_anchor  # max x, min y
        self.top_left_offset = np.array([0, 0], dtype=np.float32)
        self.bottom_right_offset = np.array([0, 0], dtype=np.float32)

        self.extents = extents
        self.extents_pixel = extents_pixel
        
    

    def _update_extents(self):
        """Recompute extents"""
        # compute (h, w) extents in meters
        self.extents = np.abs(
            self.get_top_left_anchor() - self.get_bottom_right_anchor()
        )[::-1] # invert to (h, w) convention

        # update extent dimension (anchors + offset) in pixel
        self.extents_pixel = (self.extents / self.resolution).astype(np.int32)


    def put_point(self, point: NDArray):
        """Add a point (x, y) to this region. This may change the anchors of the region
        to create a rectangular region that contains the new point at least as a corner

        Args:
            point: 2d (x,y) array representing point world coordinates in meters
        """
        # if no anchor is set, set the first point as anchors
        if self.top_left_anchor is None or self.bottom_right_anchor is None:
            self.top_left_anchor = point
            self.bottom_right_anchor = point
        
        # stack points -> (3, 2) array
        points = np.vstack((
            self.bottom_right_anchor,
            self.top_left_anchor,
            point
        ))

        # update anchors
        top_right = points.max(axis=0)  # max x, max y -> top right anchor
        bot_left = points.min(axis=0)   # min x, min y -> bottom left anchor

        self.top_left_anchor = np.array([bot_left[0], top_right[1]])        # min x, max y
        self.bottom_right_anchor = np.array([top_right[0], bot_left[1]])    # max x, min y

        self._update_extents()       

    def find(
        self,
        dataset: DataLoader,
        start_frame: int = 0,
        end_frame: int = -1
        ):
        """Find anchors of a dataset. This function is just a shortcut
        to not iterate positions and call put_point() manually

        Args:
            dataset: dataloder to navigate
            start_frame: index of te first frame of the chunk. Defaults to 0.
            end_frame: index of the last frame of the chunk. Use -1 to iterate
                untill the end of the dataset. Defaults to -1.
        """
        if end_frame == -1:
            end_frame = len(dataset)
        
        for i in range(end_frame - start_frame):    # step of 1
            i += start_frame                        # starting from start_frame
            self.put_point(dataset.position(i)[:2]) # GPS position (x,y,z)


    def set_offset(
        self,
        direction: OffsetDirection,
        value: NDArray,
        inpixel: bool = False
        ):
        """Add offsets to anchors and generate a bigger area. This is used to 
        fit bigger areas and don't lose data with a region that is too little
        and has overflowing data.


        Args:
            direction: direction where the offset is set
            value: 2d (x,y) vector representig the offset to add to the selected
                direction
            inpixel: set to true if value is mesured in pixel. Leave to false 
                if value is in meters. Defaults to False.
        """
        
        value = np.asarray(value).astype(np.float32)
        if inpixel:
            value = (value * self.resolution).astype(np.int32)
        
        if direction == OffsetDirection.LEFT_TOP:
            # to move top-left we need to subtract x (move left)
            # and add y (move up)
            self.top_left_offset = value * [-1, 1]
        elif direction == OffsetDirection.RIGHT_BOTTOM:
            # to move top-left we need to add x (move right)
            # and subtract y (move down)
            self.bottom_right_offset = value * [1, -1]

        self._update_extents()


    def get_top_left_anchor(self) -> NDArray[np.float32]:
        """Get the offsetted top left anchor in meters

        Returns:
            2d (x, y) array in meters
        """
        return self.top_left_anchor + self.top_left_offset


    def get_bottom_right_anchor(self) -> NDArray[np.float32]:
        """Get the offsetted bottom right anchor in meters

        Returns:
            2d (x, y) array in meters
        """
        return self.bottom_right_anchor + self.bottom_right_offset
    
    def get_top_right_anchor(self) -> NDArray[np.float32]:
        tl = self.get_top_left_anchor()
        br = self.get_bottom_right_anchor()

        return np.array([br[0], tl[1]])

    def get_bottom_left_anchor(self) -> NDArray[np.float32]:
        tl = self.get_top_left_anchor()
        br = self.get_bottom_right_anchor()

        return np.array([tl[0], br[1]])


    def get_center(self) -> NDArray[np.float32]:
        """Return the center in meters

        Returns:
            2d (x, y) array in meters
        """
        return np.vstack((
            self.get_top_left_anchor(),
            self.get_bottom_right_anchor()
        )).mean(axis=0)


    def size(self, inpixel: bool = False) -> NDArray:
        """Return the region size as a 2d (y, x) array
        
        (y,x) convetion is used to allow size to be directoly used as numpy 
        shapes. If you need (x,y) you can invert them using [::-1]

        Args:
            inpixel: set to true to return pixel size. Defaults to False.
            scale: scaling factor of the size. Defaults to 1.0.

        Returns:
            2d (y, x) array representing the region size in meters if inpixel 
            is false. Otherwise the returned size is mesured in pixels
        """
        return self.extents_pixel if inpixel else self.extents


    def containts(self, point: NDArray) -> bool:
        """ USE contains
        Check if this region contains a point

        Args:
            point: point in meters to check

        Returns:
            true if the point is contained in this region
        """
        return (
                # x inside bounts
                self.top_left_anchor()[0] <= point[0] and
                point[0] <= self.bottom_right_anchor()[0]
            ) and (
                # y inside bounds
                self.bottom_right_anchor()[1] <= point[1] and
                point[1] <= self.top_left_anchor()[1]
            )

    
    def to_json(self) -> Dict:
        """Convert class to json dict

        Returns:
           dict representation of this class
        """
        res = {
            "resolution": float(self.resolution),
            "anchors": {
                "top_left" : {
                    "point": self.top_left_anchor.tolist(),
                    "offset": self.top_left_offset.tolist()
                },
                "bottom_right": {
                    "point": self.bottom_right_anchor.tolist(),
                    "offset": self.bottom_right_offset.tolist()
                }
            }, 
            "extents": {
                "meter": self.extents.tolist(),
                "pixel": [int(i) for i in self.extents_pixel]
            }

        }
        return res

    @classmethod
    def from_json(cls, json):
        """Load class values from a dict/json
        """
        res = cls(
            json["resolution"],
            np.array(json["anchors"]["top_left"]["point"]),
            np.array(json["anchors"]["bottom_right"]["point"]),
            extents=np.array(json["extents"]["meter"]),
            extents_pixel=np.array(json["extents"]["pixel"], dtype=np.int32),
        )

        res.top_left_offset = np.array(
            json["anchors"]["top_left"]["offset"],
            dtype=np.float32
        )
        res.bottom_right_offset = np.array(
            json["anchors"]["bottom_right"]["offset"],
            dtype=np.float32
        )

        return res
    
    def __str__(self) -> str:
        res =  f"Extents {self.size()}m - {self.size(inpixel=True)}pixel"
        res += f"  Top Left: {self.get_top_left_anchor()}"
        res += f"  Botton Right: {self.get_bottom_right_anchor()}"
        return res

    
    def contains(self, position: NDArray) -> bool:
        """Check if a region contains a world position

        Args:
            position: (x,y) position in metes to check 

        Returns:
            true if the specified position is inside this region
        """
        tla = self.get_top_left_anchor()
        bra = self.get_bottom_right_anchor()

        x_ok = tla[0] <= position[0] and position[0] <= bra[0]
        y_ok = bra[1] <= position[1] and position[1] <= tla[1]

        return x_ok and y_ok