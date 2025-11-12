from .modelpredictor import ModelPredictor

import torch
from numpy.typing import NDArray
from ..bev import Bev
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
import cv2

class SDFPredictor(ModelPredictor):
    """Simplified wrapper of model predictor. This requires models with two outputs
    that return (line prediction, drivable prediction). Drivable prediction is assumed
    to have only 0-1 classes
    """
    def __init__(
        self,
        model,
        bev_obj: Bev,
        num_classes,
        device = "cpu",
        bev_to_forward: float = -90,
        min_block_area: float = 20,
        bev_rotation_center:tuple = None,
        predict_at = (384, 640)
        ) -> None:
        """
        Args:
            model: model to use to make predictions
            num_classes: number of classes of the model used to 
                generate predictions (including drivable)
            bev_obj: instance of Bev used to generate 
            bev_to_forward: angle in deg required to rotate the bev to look 
                at vehicle x axis,
            min_block_area: minimum area for a prediction to be coosidered v
                alid and not cleaned in postprocessing
            bev_rotation_center: center of rotation used to rotate the bev when
                generating the map
        """
        self.model = model
        # TODO: crop from center and pad later
        def pred_fn(img):
            sz = img.shape[-2:]
            img = resize(img, predict_at)

            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    img = img.unsqueeze(0).to(device) / 255.
                    line_pred, area_pred = model(img)
                    drivable_class = line_pred.shape[1]

                    line_pred = torch.argmax(line_pred, dim=1)
                    if area_pred is not None:
                        area_pred = torch.argmax(area_pred, dim=1)

                        # place drivable only where the are no lines
                        mask = line_pred == 0
                        line_pred[mask] = area_pred[mask] * drivable_class
        
                pred = line_pred.cpu()
                pred = resize(pred, sz, interpolation=InterpolationMode.NEAREST)

            return pred.squeeze().numpy()

        super().__init__(
            pred_fn,
            num_classes=num_classes,
            bev_obj = bev_obj,
            bev_to_forward = bev_to_forward,
            min_block_area=min_block_area,
            bev_rotation_center=bev_rotation_center,
            drivable_class=num_classes-1
        )
    