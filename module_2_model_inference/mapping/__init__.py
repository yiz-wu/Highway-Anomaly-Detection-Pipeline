"""Package containing the whole pipeline to transform single frame predictions
into a HD map suitable for use.
"""

from enum import Enum
class AngleMode(Enum):
    """
    Angle mode used to generate the pixel map. Note: bev rotations are 
    applied only using heading. Roll and Pitch should be included in the 
    calibration
    """

    GPS = "gps"
    """
    Use raw rotation provided by the dataloader. Use this option if you have
    good rotations and fixed bevs
    """
    ESTIMATED= "estimated"
    """
    Estimate the rotation using the last two gps positions. This option is 
    usefull when no good rotation is available.
    """
    ONLY_OFFSET = "only_offset"
    """
    Rotate only using the extra offset. This covers the case where is not 
    possible to distinguish camera calibration from vehicle rotation and the 
    resulting bevs have a different rotation at every frame
    """

class SmoothMode(Enum):
    """
    Smooth mode used to generate pixel map. This mode allows to use smoothed
    values instead of noisy ones.
    """

    NONE = "none"
    """Do not smooth anything"""
    ANGLE = "angle"
    """Use smooth angles computed by dataloader"""
    POSTION = "position"
    """Use smooth positions computed by dataloader"""
    ALL = "all"
    """Smooth both angles and positions"""

class BlendMode(Enum):
    """
    Blend mode defines how bevs are put together when generating the pixel
    map. This setting changes how the depth of the map is computed.
    """

    SUM = "sum"
    """Sum predictions channel by channel everywhere"""
    AVERAGE = "avg"
    """
    Average channels where an overlap is detected. This mode is designed to 
    be used with RGB map where you need to average overlaps to keep colors
    """
    ENHANCE_OVERLAP = "overlap"
    """
    Like sum but where an overlap is detected, the overlapping pixel are 
    weighed more. This can be used to postprocess the map and keep only pixels 
    where a lot of overlaps occurred and you can be sure of their prediction.
    """

from .camera import Camera, LadybugCamera
from .bev import Bev

from .pixelmap import PixelMap

# graph map

from .graphs import NodeConnectionMode
from .graphs import ImageParserBase, BoxImageParser

from .regioniterable import RegionIterable
from .pixelmapiterable import PixelMapIterable
from .graphmap import GraphMap

