from .dataloader import DataLoader

import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import os
from torchvision.io import read_image, ImageReadMode

class SDFLoader(DataLoader):
    """
    Simplified dataset format dataloder
    """
    def __init__(self, img_path: str, gps_path: str, img_format="png") -> None:
        super().__init__()
        self.img_path = img_path
        self.img_format = img_format
        self.start_from = 0
        self.end_at = None

        with open(gps_path, "r") as fp:
            lines = fp.readlines()[1:]

        self.sequence  = []
        self.gps_positions = []
        self.gps_rotations = []
        for line in lines:
            seq, x,y,z, yaw, roll, pitch = line.split(",")
            self.sequence.append(seq)
            self.gps_positions.append(np.array([float(x),float(y),float(z)]))
            self.gps_rotations.append(np.array([float(yaw), float(roll), float(pitch)]))
    


    def image(self, idx) -> NDArray:
        """Return image in torch format for the frame idx

        Args:
            idx: desired index

        Returns:
            image at the desired frame
        """
        idx += self.start_from
        img_name = f"{self.sequence[idx]}.{self.img_format}"
        img = read_image(os.path.join(self.img_path, img_name), ImageReadMode.RGB)

        return img

    def position(self, idx) -> NDArray:
        """Return position (x,y,z) in meters at position idx

        Args:
            idx: desired index

        Returns:
            postion in meters at the desired frame 
        """
        idx += self.start_from
        return self.gps_positions[idx]

    def rotation(self, idx) -> NDArray:
        """Return rotation (yaw, roll, pitch) in radians at position idx

        Args:
            idx: desired index

        Returns:
            rotation in radians at the desired frame
        """
        idx += self.start_from
        return self.gps_rotations[idx]
    
    def smooth_position(self, idx) -> NDArray:
        """Return smoothed position (x,y,z) in meters at position idx

        Args:
            idx: desired index

        Returns:
            smooth position in meters at the desired frame
        """
        # TODO: may be added in the future
        raise NotImplementedError

    def smooth_heading(self, idx) -> float:
        """Return smoothed yaw (vehicle rotation around up axis) in radians 
        at position idx

        Args:
            idx: desired index
        Returns:
            smooth rotation in radians at the desired frame
        """
        # TODO: may be added in the future
        raise NotImplementedError

    def restart_from_(self, start_index=0) -> None:
        """Set start position of the dataset

        Args:
            start_index: index of the first frame. Defaults to 0.
        """
        self.start_from = start_index
    
    def end_at_(self, end_index=0) -> None:
        """Set end position of the dataset

        Args:
            end_index: index of the last frame. Defaults to 0.
        """
        if end_index < 0 :
            self.end_at = len(self.sequence)
        else:
            self.end_at = np.clip(end_index,0, len(self.sequence))

    def __len__(self) -> int:
        return np.clip(len(self.sequence), 0, self.end_at) - self.start_from