import os
import re
from datetime import datetime, timedelta
from typing import List

from math import radians
import numpy as np
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.functional as TF
from scipy import interpolate

from .dataloader import DataLoader
from .utils import sorted_nicely

class AdegliaccoLoader(DataLoader):
    """Adeliacco dataset loader for both images and gps capture. This assume 
    that the frames and the gps entries are matched and in the correct order.

    Loaded images are ordered by name
    """

    def __init__(
        self,
        image_folder,
        gps_file,
        start_from=0,
        size=None,
        name_regex=None
        ) -> None:
        """Initialize dataloader

        Args:
            image_folder: path to folder containing images
            gps_file: file containing gps positions and rotations
            start_from: index of the start frame. Defaults to 0.
            size: desired image size. Defaults to None.
            name_regex: regex used to pick only matched image names.
                Defaults to None (load all).
        """
        self.img_folder = image_folder
        self.size = size
        self.gps_file = gps_file
        self.start_from = start_from

        # list of image file names
        tmp_images = os.listdir(self.img_folder)
        images = []

        # allow data filter based on regex instead of folder
        if name_regex is not None:
            prog = re.compile(name_regex)   # Pattern object
            for img in tmp_images:
                if prog.match(img) is not None:
                    images.append(img)
        else:
            images = tmp_images

        # sort image file names
        self.images = sorted_nicely(images)

        # load GPS data
        self.gps_position = [] # every element has (x, y, z) format
        self.gps_rotation = [] # every element has (yaw, roll, pitch) format

        # smoothing data (we waste a bit of ram for speed but this is not too bad)
        self.smoothed_headings = []
        self.smoothed_positions = []
        self.traj_x = []
        self.traj_y = []
        self.timestamps = []

        with open(self.gps_file, "r") as fp:
            lines = fp.readlines()
            for line in lines[1:]: # skip header line of the data file
                values = re.split(r'\t+', line)
                self.gps_position.append(
                    (
                        float(values[1]),
                        float(values[2]),
                        float(values[3])
                    )
                )
                self.gps_rotation.append(
                    (
                        radians(float(values[4])),
                        radians(float(values[5])),
                        radians(float(values[6]))
                    )
                )
                self.traj_x.append(float(values[1]))
                self.traj_y.append(float(values[2]))
                tmp = (
                    datetime.strptime(values[7][:-4], "%Y%m%d%H%M%S") +
                    timedelta(milliseconds=float(values[7][-4:]))
                )
                self.timestamps.append(tmp.timestamp())

        assert len(self.images) == len(self.gps_position),\
            f"Found {len(self.images)} images did not match the" \
            f" {len(self.gps_position)} gps positions"


    def compute_heading_smoothing(self, smoothing_mae_allowed = 0.03):
        """Compute smoothed postions and headings"""
        # fitting
        u0 = self.timestamps[0]
        w = 1e3
        u = (np.array(self.timestamps) - u0) / w
        assert len(u) == len(np.unique(u)), "U must contain all unique items"
        enu_interp, _ = interpolate.splprep([self.traj_x, self.traj_y], u=u, s=smoothing_mae_allowed * len(self.traj_x))
        enu_interp_fcn = lambda t: np.array(interpolate.splev((np.asarray(t) - u0) / w, enu_interp)).T
        enu_deriv_interp_fcn = lambda t: np.array(interpolate.splev((np.asarray(t) - u0) / w, enu_interp, der=1)).T
        heading_interp_fcn = lambda t: np.arctan2(*enu_deriv_interp_fcn((np.asarray(t)).reshape((-1, 1))[:, 0])[:, ::-1].T)

        self.smoothed_headings = heading_interp_fcn(self.timestamps)
        self.smoothed_positions = enu_interp_fcn(self.timestamps)
    
    def image(self, idx):
        idx += self.start_from
        img = read_image(os.path.join(self.img_folder, self.images[idx]), ImageReadMode.RGB)

        if self.size is not None:
            img = TF.resize(img, self.size)
        
        return img
    
    def rotation(self, idx):
        idx += self.start_from
        return np.array(self.gps_rotation[idx])
    
    def position(self, idx):
        idx += self.start_from
        return np.array(self.gps_position[idx])

    def smooth_heading(self, idx):
        idx += self.start_from
        return self.smoothed_headings[idx]
    
    def smooth_position(self, idx):
        idx += self.start_from
        return np.array(self.smoothed_positions[idx])

    def restart_from_(self, start_index=0):
        self.start_from = start_index

    def __len__(self):
        return len(self.images) - self.start_from


class Adegliacco360Loader(DataLoader):
    """
    Adeliacco dataset loader for bev generation with all 5 orizontal cameras
    """

    def __init__(
        self,
        subloaders: List[AdegliaccoLoader],
        start_from = 0
        ) -> None:
        """Initialize dataloader

        Args:
            subloaders: list of AdegliaccoDataloder to use to load data for
                every camera
            start_from: index of the start frame. Defaults to 0.
        """
        self.loaders = subloaders
        self.size = len(subloaders[0])
        self.start_from = start_from

        # check all loaders have the same size
        # and reset start position
        for l in subloaders:
            assert len(l) == self.size, "All subloaders must have the same len"
            l.restart_from_(0)
    
    def image(self, idx):
        """Return an array of images (one image per camera) for the frame idx"""
        idx += self.start_from
        images = []
        for l in self.loaders:
            images.append(l.image(idx))
        
        return images

    def rotation(self, idx):
        """Return an array of rotations (radians), one for every camera"""
        idx += self.start_from
        rotations = []
        for l in self.loaders:
            rotations.append(l.rotation(idx))
        
        return rotations
  
    
    def position(self, idx):
        """
        Return the Ladybug center in world coordinates computed as the average
        point of all camera positions. The returned center uses float64 because
        float32 can introduce errors due to big numbers killing the 
        overlall precision

        Args:
            idx: index of the desired position
        """
        idx += self.start_from
        center = np.array([0,0,0], dtype=np.float64)
        for l in self.loaders:
            center += l.position(idx)
        
        return center / len(self.loaders)

    def smooth_heading(self, idx):
        raise NotImplementedError
    
    def smooth_position(self, idx):
        idx += self.start_from
        center = np.array([0,0,0], dtype=np.float64)
        for l in self.loaders:
            center += l.smooth_position(idx)
        
        return center / len(self.loaders)

    def restart_from_(self, start_index=0):
        self.start_from = start_index

    def __len__(self):
        return self.size - self.start_from