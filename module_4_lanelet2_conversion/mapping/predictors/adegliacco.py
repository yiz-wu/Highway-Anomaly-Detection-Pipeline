from .predictor import StandardPostProcessPredictor, Predictor
import numpy as np
import cv2
from numpy.typing import NDArray
from ..bev import Bev
from ..camera import LadybugCamera
from typing import Union, List
from math import radians, atan2, degrees

from models.utils import adegliacco_rectified_predict

class Adegliacco360Predictor(Predictor):
    """Adegliacco predictor for 360 bev (use with Adegliacco360Loader)"""

    def __init__(
        self,
        model,
        cameras: List[LadybugCamera],
        resolution: float,
        num_classes: int,
        pred_vertical_offsets: List[int],
        bev_view_size: List[float] = [20,20], 
        bev_to_forward: float = -90,
        min_block_area: float = 20,
        erosion = None,
        erosion_kernel_size=(3,3)
        ) -> None:
        """
        Args:
            model: model to use to make predictions
            cameras: list of cameras to use
            resolution: resolution of the bev in meters (typically 0.05m/pixel)
            bev_view_size: bev size in meters as [horizontal size, vertical size]
            num_classes: number of classes of the model used to generate predictions
            pred_vertical_offsets: vertical offsets used to move the prediction
            qview inside the images
            bev_to_forward: angle in deg required to rotate the bev to look
                at vehicle x axis,
            min_block_area: minimum area for a prediction to be coosidered
                valid and not cleaned in postprocessing, not whole image, but connected component's prediction
            erosion: type of erosion see cv2 MorphShapes. Leave none if no 
                erosion is needed
            erosion_kernel_size: kernel size of erorion
        """
        self.model = model
        self.num_classes = num_classes
        self.bev_objs: List[Bev] = []
        self.resolution = resolution
        self.bev_to_forward = radians(bev_to_forward)
        self.pred_vertical_offsets = pred_vertical_offsets

        self.min_block_area = min_block_area
        self.erosion = erosion
        self.erosion_kernel_size = erosion_kernel_size

        x_bev = bev_view_size[0]
        y_bev = bev_view_size[1]
        for c in cameras:
            view = [-y_bev // 2, y_bev // 2, -x_bev // 2, x_bev // 2]
            self.bev_objs.append(Bev(c, view , resolution))


        self.bev_size = np.asarray(bev_view_size, dtype=np.float32) / resolution
        self.bev_rotation_center =  self.bev_size / 2


        
    def before_predicion(self, dataset, current_frame: int):
        rotations = dataset.rotation(current_frame)
        
        # recalculate camera calibration with current rotation
        for r, bev in zip(rotations, self.bev_objs):
            # the strange indexes fixes order of rotations 
            # from dataloader convention to the bev one
            bev.set_camera_rotation([r[1], r[2], r[0]])

    def predict(self, images) -> NDArray:
        preds = []
        for img, offset in zip(images, self.pred_vertical_offsets):
            pred = adegliacco_rectified_predict(
                self.model,
                img,
                vertical_view_offset=offset
            )
            preds.append(pred)
        
        return preds


    def predict_bev(self, images) -> NDArray:
        """Make prediction on camera image and return the corresponding projected bird eye view."""
        preds = self.predict(images)
        bev_out = None

        for pred, bev_obj in zip(preds, self.bev_objs):
            bev = bev_obj.computeBev(pred, cv2.INTER_NEAREST).astype(np.uint8)
            # convert into onehot encoding to make the sum of preditions easy
            # this also allows to have more robust outputs when multiple cameras
            # that overlap predict the same output for a pixel
            bev = (np.arange(self.get_map_depth()) == bev[...,None])
            # The expression (np.arange(self.get_map_depth()) == bev[..., None]) performs element-wise comparison, resulting in a one-hot encoded array where each pixel is represented by a binary vector of length 12.
            # For example, if a pixel value in bev is 3, the corresponding one-hot encoded vector will be [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0].
            # If bev = np.array([[0, 1], [2, 3]]), then the one-hot encoded array will be: bev = np.array([ [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]] ])

            if bev_out is None:
                bev_out = bev
            else:
                bev_out += bev
        
        return bev_out
    
    def post_process_bev(self, img: NDArray) -> NDArray:
        if self.min_block_area is None:
            return img
            
        tmp = np.copy(img)

        drivable_class = self.num_classes-1 # [0 is background, 1-10 are classes, 11 is drivable area]
        tmp[..., 0] = 0                 # remove background
        tmp[..., drivable_class] = 0    # remove drivable area

        # Create a binary image from the input image, where 0 is background and 1 is any class
        src = (tmp.sum(axis=-1) != 0).astype(np.uint8)

        # search components
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(src, connectivity=8)
        sizes = stats[1:, -1]   # size of each components except background 0
        nb_components = nb_components - 1 
        #keep components bigger than required size
        keep = np.zeros((img.shape[:2]))
        for j in range(0, nb_components):
            if sizes[j] >= self.min_block_area:
                keep[output == j + 1] = 1 
        
        if self.erosion is not None:
            element = cv2.getStructuringElement(self.erosion, self.erosion_kernel_size)
            keep = (cv2.erode(keep, element, borderType=cv2.BORDER_CONSTANT) + 0.5).astype(np.uint8)

        #print(f"Keep {keep.sum()} of {img.shape[0] * img.shape[1]}")

        # virtually repeat weights for every class
        keep = np.broadcast_to(keep[...,None], keep.shape + (self.num_classes, )) 
    
        res = keep * img
        # add back the drivable & background
        res[..., 0] = img[..., 0]
        res[..., drivable_class] = img[..., drivable_class]

        return np.uint8(res)

    def get_bev_to_forward_angle(self) -> float:
        return self.bev_to_forward

    def get_bev_center(self) -> Union[tuple, None]:
        return self.bev_rotation_center   

    def get_map_depth(self) -> float:
        return self.num_classes

    def get_bev_size(self) -> NDArray:
        return self.bev_size.astype(np.int32)

    def get_map_resolution(self) -> float:
        return self.resolution



class AdegliaccoRGB360Predictor(Predictor):
    """Adegliacco RGB predictor for 360 bev (use with Adegliacco360Loader)"""
    def __init__(
        self,
        cameras: List[LadybugCamera],
        resolution: float,
        bev_view_size: List[float] = [20,20], 
        bev_to_forward: float = -90,
        ) -> None:
        """
        Args:
            cameras: list of cameras to use
            resolution: resolution of the bev in meters (typically 0.05m/pixel)
            bev_view_size: bev size in meters as [horizontal size, vertical size]
            bev_to_forward: angle in deg required to rotate the bev to look at
            vehicle x axis,
        """
        self.bev_objs: List[Bev] = []
        self.resolution = resolution
        self.bev_to_forward = radians(bev_to_forward)

        x_bev = bev_view_size[0]
        y_bev = bev_view_size[1]
        for c in cameras:
            view = [-y_bev // 2, y_bev // 2, -x_bev // 2, x_bev // 2]
            self.bev_objs.append(Bev(c, view, resolution))


        self.bev_size = np.asarray(bev_view_size, dtype=np.float32) / resolution
        self.bev_rotation_center =  self.bev_size / 2

        self.car_mask = np.ones(self.bev_size.astype(np.int32), dtype=np.uint8)
        v_off = 0
        self.car_mask[100+v_off:200 + v_off, 165:250, ...] = 0
        self.car_mask[50+v_off:100 + v_off, 170:200 , ...] = 0

        self.rotated_car_mask = None
        
    def before_predicion(self, dataset, current_frame: int):
        rotations = dataset.rotation(current_frame)
        
        # recalculate camera calibration with current rotation
        for r, bev in zip(rotations, self.bev_objs):
            # the strange indexes fixes order of rotations from dataloader 
            # convention to the bev one
            bev.set_camera_rotation([r[1], r[2], r[0]])

        # estimate angle to rotate car mask
        prev_frame = current_frame -1 if current_frame - 1 >= 0 else len(dataset) -1
        delta = dataset.position(current_frame) - dataset.position(prev_frame)
        angle = degrees(atan2(delta[1], delta[0])) + self.bev_to_forward
        R = cv2.getRotationMatrix2D(self.bev_rotation_center, angle, 1.0)
        self.rotated_car_mask = cv2.warpAffine(self.car_mask, R, self.get_bev_size()[::-1])


    def predict(self, images) -> NDArray:
        preds = []
        for img in images:
            img = img.permute(1,2,0).numpy().astype(np.uint8)
            #mask overflow region (half-left of image)
            img[:, 0: int(img.shape[1] * 0.5), : ] = 0
            preds.append(img)
        
        return preds


    def predict_bev(self, images) -> NDArray:
        """Make prediction and return the corresponding bird eye view."""
        preds = self.predict(images)
        bev_out = None

        for pred, bev_obj in zip(preds, self.bev_objs):
            bev = bev_obj.computeBev(pred, cv2.INTER_LINEAR).astype(np.uint8)

            if bev_out is None:
                bev_out = bev
            else:
                # blend overlap by using a weighed sum
                overlap = (bev_out.sum(-1) != 0) * (bev.sum(-1) != 0)
                if overlap.sum() > 1:
                    bev_out[overlap] = cv2.addWeighted(
                        bev_out[overlap],
                        0.5,
                        bev[overlap],
                        0.5,
                        0
                    )
                    bev[overlap] = 0

                bev_out += bev

                # multiply by rotated car mask 
                # this is not perfect, but should be good enough
                alpha = np.broadcast_to(
                    self.rotated_car_mask[...,None],
                    self.rotated_car_mask.shape + (3, )
                )
                bev_out *= alpha
        
        return bev_out.astype(np.uint8)
    
    def get_bev_to_forward_angle(self) -> float:
        return self.bev_to_forward

    def get_bev_center(self) -> Union[tuple, None]:
        return self.bev_rotation_center   

    def get_map_depth(self) -> float:
        return 3

    def get_bev_size(self) -> NDArray:
        return self.bev_size.astype(np.int32)

    def get_map_resolution(self) -> float:
        return self.resolution