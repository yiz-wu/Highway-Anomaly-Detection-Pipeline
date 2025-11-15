# generated both segmentation and bev map from a  dataset in sdf format

import sys
import os
sys.path.append(os.path.abspath("./"))

import json
import argparse
import inspect
import numpy as np

def name2class_map(module):
    members = inspect.getmembers(module, lambda x: inspect.isclass(x) or inspect.isfunction(x))
    return {name: obj for name, obj in members}

################################################################################
# Command args

parser = argparse.ArgumentParser(description='Generate pixel/rgb map from json preset')
parser.add_argument(
    "-i",
    dest="config_file",
    required=True,
    help="Path to JSON preset configuration file"
)
args = parser.parse_args()

json_path = str(args.config_file)
if not os.path.exists(json_path):
    print(f"Unable to locate file {json_path}")
    exit(1)
with open(json_path, "r") as fp:
    preset = json.load(fp)

mode = preset.get("mode")
print(f"Using preset: {json_path} - {mode} mode")

################################################################################
# Prepare dataloder

from mapping.dataloader import SDFLoader

dataset_cfg = preset.get("dataset")

car_gps_path = dataset_cfg.get("car_gps")
images_folder = dataset_cfg.get("images")
img_format = dataset_cfg.get("img_format")
name_regex = dataset_cfg.get("name_regex")
start_idx = dataset_cfg.get("start", None)
end_idx = dataset_cfg.get("end", None)

interpolated_path = os.path.join(os.path.dirname(car_gps_path), f"interpolated_{os.path.basename(car_gps_path)}")
if os.path.exists(interpolated_path):
    car_gps_path = interpolated_path

dataloader = SDFLoader(
    images_folder,
    car_gps_path,
    img_format,
    name_regex
)

if start_idx is not None:
    dataloader.restart_from_(start_idx)
if end_idx is not None:
    dataloader.end_at_(end_idx)

print(f"Found {len(dataloader)} frames")

################################################################################
# Prepare camera and bev
from mapping import Bev
import mapping.camera as mca

valid_cameras = name2class_map(mca)

bev_cfg = preset.get("bev")
offset_angle = bev_cfg.get("offset_angle", 0)
center_of_rotation = bev_cfg.get("center_of_rotation")
mask_bev = bev_cfg.get("mask_bev")
view_size = bev_cfg.get("view_size")
resolution = bev_cfg.get("resolution")

cam_cfg = bev_cfg.get("camera")
cam_name = cam_cfg.get("name")
cam_parameters = cam_cfg.get("parameters")
if cam_parameters is None:
    raise ValueError("Camera configuration missing required 'parameters' section.")

cam = valid_cameras[cam_name](**cam_parameters)
bev_obj = Bev(cam, view_size, resolution)



################################################################################
# Load Model and generate predictor
from mapping.predictors import RGBPredictor, SDFPredictor

predictor = None
if mode == "class":
    import torch
    import models

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        gpu = torch.cuda.get_device_name(torch.cuda.current_device())
        print("Found GPU: ", gpu)
        model_device = torch.device("cuda")
    else:
        print("Using CPU")
        model_device = torch.device("cpu")

    model_cfg = preset.get("model")
    custom_model_path = model_cfg.get("path","/app/custom_models")
    model_name = model_cfg.get("name")
    model_parameters = model_cfg.get("parameters")
    if model_parameters is None:
        raise ValueError("Model configuration missing required 'parameters' section.")
    model_checkpoint = model_cfg.get("checkpoint")
    model_num_classes = model_cfg.get("num_classes")
    model_predict_size = model_cfg.get("predict_size", [384, 640])

    valid_modes = name2class_map(models)
    valid_modes.update(models.load_custom_models(custom_model_path))

    model = valid_modes[model_name](**model_parameters)
    # support jit or other strange loading inside warped model constructors

    if model_checkpoint:
        model.load_state_dict(torch.load(model_checkpoint, map_location=model_device))
    model.to(model_device)
    model.eval()

    predictor = SDFPredictor(
        model,
        bev_obj,
        model_num_classes,
        device=model_device,
        bev_to_forward=offset_angle,
        bev_rotation_center=tuple(center_of_rotation),
        predict_at=tuple(model_predict_size)
    )
elif mode == "RGB":
    # car masking is not supported in this simplified mode
    predictor = RGBPredictor(
        bev_obj,
        offset_angle,
        tuple(center_of_rotation)
    )

################################################################################
# Bev mask - works only for non rotating and non 360 cameras

bev_mask = None

if mask_bev:
    tmp = RGBPredictor(
        bev_obj,
        offset_angle,
        tuple(center_of_rotation)
    )
    timg, _, _ = dataloader[int(len(dataloader)/2)]
    bev_mask = tmp.predict_bev(timg).sum(-1) > 0
    bev_mask = bev_mask.astype(np.int32)
    # a [1,0] mask of dimension of bev used to identify the area projected by images, other areas are empty regions/padding of BEV


################################################################################
# Generate pixel map
import shutil
from mapping import PixelMap, BlendMode, SmoothMode

map_preset = preset.get("pixel-mapping")

out_name = "PixelMap" if mode == "class" else "PixelMapRGB"
out_path = os.path.join(preset.get("output_root", "/app/output"), out_name)

# clear old map
if os.path.exists(out_path):
    shutil.rmtree(out_path)

# create and save map from dataset
map = PixelMap()
map.from_dataset(
    predictor,
    dataloader,
    save_path=out_path,
    distance_weigh_mask = bev_mask,
    blend_mode=BlendMode.ENHANCE_OVERLAP if mode == "class" else BlendMode.AVERAGE,
    smooth_mode=SmoothMode.NONE,
    **map_preset.get("parameters")
)

################################################################################
# Generate chunk images if required

# this script does not generate full dataset images because that may
# crash your machine due to extreme ram usage

if map_preset.get("visualization").get("saveimages"):
    print("Saving chunk images...")
    img_folder = map_preset.get("visualization").get("folder")

    save_path = os.path.join(
        preset.get("output_root", "/app/output"),
        img_folder,
        out_name,
        "Chunks",
    )

    os.makedirs(save_path, exist_ok=True)

    if mode == "class":
        map.save_chunks_rgb(save_path)
    else:
        import cv2

        for c in map.chunks:
            dm = c.dense()
            output = cv2.cvtColor(dm, cv2.COLOR_RGB2BGR)
            name = f"{c.start_frame}-{c.end_frame}.png"
            cv2.imwrite(os.path.join(save_path, name), output) 
    
    print(f"Saved images at {save_path}")
