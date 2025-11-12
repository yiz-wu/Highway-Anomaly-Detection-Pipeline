from .predictor import StandardPostProcessPredictor, Predictor
import numpy as np
import cv2
from numpy.typing import NDArray
from ..bev import Bev
from typing import Union
from math import radians

class ModelPredictor(StandardPostProcessPredictor):
    """Generic model predictor that can be used with segmentation models"""

    def __init__(
        self,
        pred_fn,
        num_classes,
        bev_obj: Bev,
        bev_to_forward: float = -90,
        min_block_area: float = 20,
        bev_rotation_center:tuple = None,
        drivable_class:int = None,
        erosion = None,
        erosion_kernel_size=(3,3)
        ) -> None:
        """
        Args:
            pred_fn: function that returns the prediction for and image as 
                integer labels
            num_classes: number of classes of the model used to 
                generate predictions
            bev_obj: instance of Bev used to generate 
            bev_to_forward: angle in deg required to rotate the bev to look 
                at vehicle x axis,
            min_block_area: minimum area for a prediction to be coosidered v
                alid and not cleaned in postprocessing
            bev_rotation_center: center of rotation used to rotate the bev when
                generating the map
            drivable_class: integer indicating the drivable class, if none is 
                passed no drivable class is considered in postprocessing
        """
        super().__init__(
            min_block_area,
            drivable_class,
            erosion=erosion,
            erosion_kernel_size=erosion_kernel_size
        )
        self.pred_fn = pred_fn
        self.num_classes = num_classes
        self.bev_obj = bev_obj
        self.bev_to_forward = radians(bev_to_forward)

        if bev_rotation_center is not None:
            self.bev_rotation_center = bev_rotation_center
        else:
            px = bev_obj.bev_shape()[1]
            py = bev_obj.bev_shape()[0]
            self.bev_rotation_center = (int(px/2), py)

    def predict(self, img) -> NDArray:
        return self.pred_fn(img)

    def predict_bev(self, img) -> NDArray:
        pred = self.predict(img).astype(np.uint8)
        return self.bev_obj.computeBev(pred, cv2.INTER_NEAREST)

    def get_bev_to_forward_angle(self) -> float:
        return self.bev_to_forward

    def get_bev_center(self) -> Union[tuple, None]:
        return self.bev_rotation_center
        

    def get_map_depth(self) -> float:
        return self.num_classes

    def get_bev_size(self) -> NDArray:
        return np.asarray(self.bev_obj.bev_shape())[:2]

    def get_map_resolution(self) -> float:
        return self.bev_obj.get_resolution()


class SoftmaxPredictor(Predictor):
    """
        Generic model predictor that can be used to return softmaxed outputs of a newtwork
        This predictor has no postprocessing
    """

    def __init__(
        self,
        pred_fn,
        num_classes,
        bev_obj: Bev,
        bev_to_forward: float= -90,
        bev_rotation_center:tuple = None,
        ) -> None:
        """
        Args:
            pred_fn: function that returns the prediction for and image as integer labels
            num_classes: number of classes of the model used to generate predictions
            bev_obj: instance of Bev used to generate 
            bev_to_forward: angle in deg required to rotate the bev to look ast vehicle x axis,
            bev_rotation_center: center of rotation used to rotate the bev when generating the map
        """
        super().__init__()
        self.pred_fn = pred_fn
        self.num_classes = num_classes
        self.bev_obj = bev_obj
        self.bev_to_forward = bev_to_forward

        if bev_rotation_center is not None:
            self.bev_rotation_center = bev_rotation_center
        else:
            px = bev_obj.bev_shape()[1]
            py = bev_obj.bev_shape()[0]
            self.bev_rotation_center = (int(px/2), py)

    def predict(self, img) -> NDArray:
        return self.pred_fn(img)

    def predict_bev(self, img) -> NDArray:
        """Make prediction and return the corresponding bird eye view."""
        pred = self.predict(img)
        return self.bev_obj.computeBev(pred, cv2.INTER_NEAREST)

    def get_bev_to_forward_angle(self) -> float:
        """Return angle (radians) required to make this bev point in the forward direction of the vehicle (usually positive x axis)"""
        return self.bev_to_forward

    def get_bev_center(self) -> Union[tuple, None]:
        """Return bev center of rotation. If None is returned bottom center is used"""
        return self.bev_rotation_center
        

    def get_map_depth(self) -> float:
        """Return map depth"""
        return self.num_classes

    def get_bev_size(self) -> NDArray:
        """Return bev size in pixel. this should be a 2D array"""
        return np.asarray(self.bev_obj.bev_shape())[:2]

    def get_map_resolution(self) -> float:
        """Return map resolution in m/pixel"""
        return self.bev_obj.get_resolution()
