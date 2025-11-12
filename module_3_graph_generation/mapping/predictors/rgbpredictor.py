from pickletools import uint8
from .predictor import Predictor
# import torch
import numpy as np
from math import radians
from numpy.typing import NDArray
from ..bev import Bev
from typing import Union

class RGBPredictor(Predictor):
    """Generic predictor usefull to stitch rgb bevs (requires torch and torch 
    dataloader)
    """

    def __init__(
        self,
        bev_obj: Bev,
        bev_to_forward: float = -90,
        bev_rotation_center:tuple = None,
        prediction_mask = None,
        image_mask = None
        ) -> None:
        """
        Args:
            bev_obj: instance of Bev used to generate 
            bev_to_forward: angle in deg required to rotate the bev to look 
                at vehicle x axis,
            bev_rotation_center: center of rotation used to rotate the bev
                when generating the map
            prediction_mask: 2d 1/0 mask for the bev to hide unwanted pixels 
            image_mask: 2d 1/0 mask multiplied to the image before generating 
                the bev
        """
        super().__init__()
        self.bev_obj = bev_obj
        self.bev_to_forward = radians(bev_to_forward)

        if bev_rotation_center is not None:
            self.bev_rotation_center = bev_rotation_center
        else:
            px = bev_obj.bev_shape()[1]
            py = bev_obj.bev_shape()[0]
            self.bev_rotation_center = (int(px/2), py)

        self.alpha = None
        if prediction_mask is not None:
            self.alpha = np.broadcast_to(
                prediction_mask[...,None],
                prediction_mask.shape + (3, )
            ) 

        if image_mask is not None:
            self.img_alpha = np.broadcast_to(
                image_mask[...,None], image_mask.shape + (3, )
            )
        else:
            self.img_alpha = None

    def predict(self, img) -> NDArray:
        img = img.permute(1,2,0).numpy().astype(np.float32)

        if self.img_alpha is not None:
            img *= self.img_alpha

        return img.astype(np.uint8)

    def predict_bev(self, img) -> NDArray:
        pred = self.predict(img)
        bev = self.bev_obj.computeBev(pred)
        if self.alpha is not None:
            bev *= self.alpha
            bev = bev.astype(np.uint8)

        return bev

    def post_process_bev(self, img: NDArray) -> NDArray:
        return img

    def get_bev_to_forward_angle(self) -> float:
        return self.bev_to_forward

    def get_bev_center(self) -> Union[tuple, None]:
        return self.bev_rotation_center
        

    def get_map_depth(self) -> float:
        return 3

    def get_bev_size(self) -> NDArray:
        return np.asarray(self.bev_obj.bev_shape())[:2]

    def get_map_resolution(self) -> float:
        return self.bev_obj.get_resolution()
