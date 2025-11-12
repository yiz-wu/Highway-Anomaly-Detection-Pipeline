"""Utility functions to work available models
"""

import torch
import cv2
import numpy as np

def count_parameters(model):
    """Count parameters of a torch model

    Args:
        model: torch model

    Returns:
        parameter count of model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_parameter_count(model):
    """Print parameter count of model

    Args:
        model: torch model
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    print(f"Total parameters: {trainable + non_trainable :,}"\
        f"\nTrainable parameters: {trainable:,} - Non trainable: {non_trainable:,}"
    )

def merge_predictions(line_pred, area_pred):
    """Merge two prediction, give preference to line_prediction
    
    Args:
        line_pred: line prediction output of a model
        area_pred: area/drivable prediction output of a model

    Returns:
        prediction with both lines and drivable area

    """
    drivable_class = line_pred.shape[1] # 11
    line_pred = torch.argmax(line_pred, dim=1)  # get highest prob. class for each pixel
    area_pred = torch.argmax(area_pred, dim=1)  # get highest prob. class for each pixel

    # place drivable only where the are no lines
    mask = line_pred == 0
    line_pred[mask] = area_pred[mask] * drivable_class
    return line_pred.squeeze()

def adegliacco_rectified_predict(
    model,
    image,
    include_drivable = True,
    vertical_view_offset:int = 0,
    horizontal_view_offset:int = 450,
    view_height:int = 384,
    view_width:int = 1152,
    return_crop = False,
    device = "cuda"
    ):
    """
    Make a prediction for a rectified Adegliacco image that is rotated -90 deg 
    and has a resolution of 2448x2048. The prediction is made on the image 
    rotated vertical and using an horizontal window of height view_height and 
    width view_width moved down by from image center by vertical_view_offset
    and left by horizontal_view_offset

    It's recommend to change only vertical_view_offset depending on the camera.
    Other paramters should not be modified

    Args:
        model: model used, should return a tuple 
            (line prediction, drivable prediction)
        image: rgb image in torch format (3, h, w)
        include_drivable: return drivable in the output
        vertical_view_offset: vertical offset of the precition view starting 
            from vertical image center (rotated image)
        horizontal_view_offset: horinztaol offset of the precition view starting
            from left (rotated image)
        view_height: prediction view height in pixel
        view_width: prediction view width in pixel
        return_crop: set to true to make this function return the image crop and
            cropped prediction. This is useful for debug or visualization 
            purposes

    Returns:
        2d prediction for every pixel in the image.
        The ouput size is (2048, 2448)

        if return_crop is enabled a tuple of (crop, crop_prediction) is returned
        instead of the 2d prediction of the whole image
    """
    image = image / 255.
    sz = image.shape #torch: (c, h, w)

    v_view_start = 2448 - 1280 + vertical_view_offset
    h_view_start = horizontal_view_offset

    image = image[
        :,
        h_view_start : h_view_start + view_width,
        v_view_start : v_view_start + view_height
        ]

    # rotate 90 clockwise
    # btw torch is bad and makes a copy when flipping...
    image = torch.flip(torch.swapaxes(image, -2, -1), dims=(-1,))

           
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            line_pred, area_pred = model(image.unsqueeze(0).to(device))
            drivable_class = line_pred.shape[1]

            line_pred = torch.argmax(line_pred, dim=1)
            if area_pred is not None:
                area_pred = torch.argmax(area_pred, dim=1)

                # place drivable only where the are no lines
                mask = line_pred == 0
                line_pred[mask] = area_pred[mask] * drivable_class
        
        pred = line_pred.squeeze().cpu()

        if not include_drivable:
            pred[pred == drivable_class] = 0     
    
    if return_crop:
        return (image * 255., pred)
    
    original_size = np.zeros(sz[-2:])

    # rotate back and set inside expected image size
    original_size[
        h_view_start : h_view_start + view_width,
        v_view_start : v_view_start + view_height
    ] = pred.numpy()[...,::-1].swapaxes(-2,-1)
    
    return original_size
