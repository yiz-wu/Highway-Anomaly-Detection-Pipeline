from typing import Union, Tuple, List
import math

import numpy as np
import cv2
from numpy.typing import NDArray

from shapely.geometry import Polygon, Point

################################################################################
# list utils

def valid_list_append(list1, list2):
    """Append list2 to list1. If one of them is None or has 0 elements is skipped.
    If both are None: None is returned.
    if both are empy an empy list is returned
    """
    if list1 is None and list2 is None:
        return None
    if list1 is None or len(list1) <= 0:
        return list2
    if list2 is None or len(list2) <= 0:
        return list1
    
    return list1 + list2

################################################################################
# coordinate utils

def pixel2worldcoord(points: NDArray, anchor: NDArray, resolution: float):
    """Convert pixel position to world coordinates

    Args:
        points: poitins to translate
        anchor: top left anchor of the image
        resolution: image resolution in m/pixel

    Returns:
        points in world coordinates
    """
    points = points * [1, -1] * resolution
    points = points + anchor
    return points

################################################################################
# img utils

def rotate_image(img, rotation, center) -> Union[NDArray, NDArray]:
    """Rotate and image and return the rotation matrix used
    
    If rotation is None nothing happens and a None rotation matrix is returned
    """
    rot_matrix = None

    if rotation is not None : 
        rot_matrix = cv2.getRotationMatrix2D(
            center,
            math.degrees(rotation),
            1.0
        )
        img = cv2.warpAffine(
            img,
            rot_matrix,
            img.shape[:2][::-1],
            flags=cv2.INTER_NEAREST
        )

    return img, rot_matrix

def rotation_matrix2D(rotation) -> NDArray:
    """Compute 2d rotation matrix

    Args:
        rotation: rotation in radians

    Returns:
        2x2 rotation matrix
    """
    rot = cv2.getRotationMatrix2D((0,0), math.degrees(rotation), 1)
    return rot[:, :2]

################################################################################
# Rect utils

def is_overlaping2D(box1: NDArray, box2: NDArray) -> bool:
    """Check if two boxes overlap given their corner postions

    Box shapes must be: (4, 2)

    Args:
        box1: first box to check
        box2: second box to check

    Returns:
        true if two box overlap
    """ 
    pbox1 = Polygon(box1)
    pbox2 = Polygon(box2)

    return pbox2.overlaps(pbox1)

def is_inside_box(point: NDArray, box: NDArray) -> bool:
    """Check if a point is contained into a box

    Box shapes must be: (4, 2)

    Args:
        point: point to check
        box: box coordinates

    Returns:
        true if the point is contained in the box
    """

    # https://stackoverflow.com/questions/57965493/how-to-convert-numpy-arrays-obtained-from-cv2-findcontours-to-shapely-polygons
    poly = Polygon(box)

    return poly.contains(Point(point))

def is_box_overlapping_percent(box1, box2, percent):
    """Check that box2 overlaps at least percent% area of box1"""
    pbox1 = Polygon(box1)
    pbox2 = Polygon(box2)

    union = pbox1.intersection(pbox2)

    return union.area / pbox1.area > percent

def split_horizontal_rect(
    rect: Tuple,
    width_pixels: float,
    overlap_pixels: float
    ) -> List[Tuple]:
    """Divide an horizontal cv2 rect into little ovelapping rects

    Args:
        rect: cv2 rect to split
        width_pixels: size of every subsplit
        overlap_pixels: number of overlapping pixels between the generated rects

    Returns:
        list or rotated rects in cv2 format
    """
    #TODO: may be better to weigh center position based on pixel inside square

    RB = cv2.getRotationMatrix2D((0,0), -rect[2], 1)[:, :2]
    
    half_size = np.array(rect[1]) / 2

    subrects = int(abs(rect[1][1]) / (width_pixels - overlap_pixels))

    center_offset = abs(rect[1][1]) / subrects 

    offsets = np.array([center_offset * (i + 1) for i in range(subrects)])
    offsets = half_size[1] - offsets + center_offset / 2
    new_centers = np.vstack((np.zeros_like(offsets), offsets))
    new_centers = rect[0] + (RB @ new_centers).T

    sz = (rect[1][0], center_offset + overlap_pixels)
    return [(tuple(center), sz, rect[2]) for center in new_centers]