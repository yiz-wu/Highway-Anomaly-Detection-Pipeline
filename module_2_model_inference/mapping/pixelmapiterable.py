from typing import List, Tuple
import math
import numpy as np
from numpy.typing import NDArray

from .pixelmap import PixelMap
from .dataloader import DataLoader

from .regioniterable import RegionIterable

class PixelMapIterable(RegionIterable):
    """Iterator designed to move around a pixel map based on dataloader positions
    """

    def __init__(
        self,
        map: PixelMap, 
        dataloader: DataLoader,
        region_size: List[float] = [20,20],
        ignore_classes: List[int] = [],
        rotation_offset = -90,
        return2d = False,
        min_area_size = None,
        ):
        """
        Args:
            map: map to iterate
            dataloder: dataloader used to generate the map
            region_size: region size in meters taken around every position of the
                dataloder. Defaults to [20,20].
            ignore_classes: List of class indexes to ignore. Defaults to [].
            rotation_offset: extra rotation in degrees added to freely rotate 
                output regions and best align them for parsing. Default to -90
            return2d: return 2d map instead of 3d one,
            min_area_size: set a minimum area used to clean junk at chunk level,
                Defaults to None (disabled)
        """
        super().__init__()
        self.map = map
        self.dataloader = dataloader
        self.region_size= np.asarray(region_size)
        self.ignore_classes = ignore_classes
        self.rotation_offset = rotation_offset

        self.rotation_center = self.region_size / 2 / self.map.get_resolution()
        self.rotation_center = self.rotation_center.astype(np.int32)

        self.return2d = return2d

        self.min_area_size = min_area_size

    def __angle_rad(self, delta):
            return math.atan2(delta[1], delta[0])

    def center_of_rotation(self) -> NDArray:
        """Return center of rotation in pixels"""
        return self.rotation_center.tolist()

    def get_region_around(
        self,
        frame,
        drivable_weight=.2,
        bg_weight=.1,
        drivable_class=-1
        ):
        center_pos = self.dataloader.position(frame)[:2]

        area, top_left_ancor = self.map.region_around(
            center_pos,
            self.region_size,
            autofix=True,
            min_area_size=self.min_area_size,
            drivable_class=drivable_class
        )

        # process area depth map to a readable and usable format

        area[:,:,0] = bg_weight # fix background class weight
        if drivable_class is not None:
            area[:,:,drivable_class] *=  drivable_weight # reduce drivable weight

        if not self.return2d:
            prd = np.argmax(area, axis=-1)
            for c in self.ignore_classes:
                area[prd==c] = 0
            area[:,:,0] = .1
        else:
            area = np.argmax(area, axis=-1)
            for c in self.ignore_classes:
                area[area == c] = 0

        angle = self.__angle_rad(
            center_pos - self.dataloader.position(frame-1)[:2]
        )

        return area, top_left_ancor, center_pos, angle


    def __len__(self) -> int:
        return len(self.dataloader)


    def __getitem__(self, idx) -> Tuple[NDArray, NDArray, List[int], float, float]:
        area, tl_anchor, center_anchor, angle = self.get_region_around(idx)

        # compute center in pixel (may change due to region autofix)
        pixel_center = (center_anchor - tl_anchor) * [1, -1] / self.map.get_resolution()
        #center_anchor = top_left + [1, -1] * self.region_size / 2

        # align forward to positive y
        to_vertical_angle = - angle - math.radians(self.rotation_offset) 

        return (
            area,
            center_anchor,
            pixel_center.astype(np.int32).tolist(),
            to_vertical_angle,
            -to_vertical_angle
        )