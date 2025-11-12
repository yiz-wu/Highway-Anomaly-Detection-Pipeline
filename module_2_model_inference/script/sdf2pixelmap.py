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
    default="monza_11_mapping.config.json",
    help="Json preset file path"
)

args = parser.parse_args()
json_path = str(args.config_file)

if not os.path.exists(json_path):
    print(f"Unable to locate file {json_path}")
    exit(1)

with open(json_path, "r") as fp:
    preset = json.load(fp)

mode = preset["mode"]
print(f"Using preset: {json_path} - {mode} mode")

################################################################################
# Prepare dataloder

from mapping.dataloader import SDFLoader

dt_preset = preset["dataset"]

car_gps_path = dt_preset["car_gps"]
interpolated_path = os.path.join(os.path.dirname(car_gps_path), f"interpolated_{os.path.basename(car_gps_path)}")
if os.path.exists(interpolated_path):
    car_gps_path = interpolated_path

dataloader = SDFLoader(
    dt_preset["images"],
    car_gps_path,
    dt_preset["img_format"] if "img_format" in dt_preset else "png",
    name_regex=dt_preset.get("name_regex", None)
)

start_idx = dt_preset.get("start", None)
if start_idx is not None:
    dataloader.restart_from_(start_idx)

end_idx = dt_preset.get("end", None)
if end_idx is not None:
    dataloader.end_at_(end_idx)

print(f"Found {len(dataloader)} frames")

################################################################################
# Prepare camera and bev
from mapping import Bev
import mapping.camera as mca


valid_cameras = name2class_map(mca)

bev_preset = preset["bev"]
cam_preset = bev_preset["camera"]
cam = valid_cameras[cam_preset["name"]](**cam_preset["parameters"])

bev_obj = Bev(cam, bev_preset["view_size"], bev_preset["resolution"])


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

    model_preset = preset["model"]

    valid_modes = name2class_map(models)
    valid_modes.update(models.load_custom_models(model_preset.get("path","/app/custom_models")))

    model = valid_modes[model_preset["name"]](**model_preset["parameters"])
    # support jit or other strange loading inside warped model constructors

    if "checkpoint" in model_preset and model_preset["checkpoint"] is not None:
        model.load_state_dict(torch.load(model_preset["checkpoint"],map_location=model_device))
    model.to(model_device)
    model.eval()

    predictor = SDFPredictor(
        model,
        bev_obj,
        model_preset["num_classes"],
        device=model_device,
        bev_to_forward=bev_preset["offset_angle"] if "offset_angle" in bev_preset else 0,
        bev_rotation_center=tuple(bev_preset["center_of_rotation"]),
        predict_at=tuple(model_preset["predict_size"]) if "predict_size" in model_preset else (384, 640)
    )
elif mode == "RGB":
    # car masking is not supported in this simplified mode
    predictor = RGBPredictor(
        bev_obj,
        bev_preset["offset_angle"] if "offset_angle" in bev_preset else 0,
        tuple(bev_preset["center_of_rotation"])
    )

################################################################################
# Bev mask - works only for non rotating and non 360 cameras

bev_mask = None

if "mask_bev" in bev_preset and bev_preset["mask_bev"]:
    tmp = RGBPredictor(
        bev_obj,
        bev_preset["offset_angle"] if "offset_angle" in bev_preset else 0,
        tuple(bev_preset["center_of_rotation"])
    )
    timg, _, _ = dataloader[int(len(dataloader)/2)]
    bev_mask = tmp.predict_bev(timg).sum(-1) > 0
    bev_mask = bev_mask.astype(np.int32)
    # a [1,0] mask of dimension of bev used to identify the area projected by images, other areas are empty regions/padding of BEV


################################################################################
# Generate pixel map
import shutil
from mapping import PixelMap, BlendMode, SmoothMode

map_preset = preset["pixel-mapping"]

out_name = "PixelMap" if mode == "class" else "PixelMapRGB"
out_path = os.path.join(preset["output_root"], out_name)

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
    **map_preset["parameters"]
)

################################################################################
# Generate chunk images if required

# this script does not generate full dataset images because that may
# crash your machine due to extreme ram usage

if map_preset["visualization"]["saveimages"]:
    print("Saving chunk images...")
    img_folder = map_preset["visualization"]["folder"]

    save_path = os.path.join(
        preset["output_root"],
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
