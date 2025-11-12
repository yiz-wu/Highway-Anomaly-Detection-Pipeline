from typing import Tuple
from numpy.typing import NDArray

class DataLoader:
    """
    Base class for all dataloaders
    """

    def image(self, idx) -> NDArray:
        """Return image in torch format for the frame idx

        Args:
            idx: desired index

        Returns:
            image at the desired frame
        """
        raise NotImplementedError

    def position(self, idx) -> NDArray:
        """Return position (x,y,z) in meters at position idx

        Args:
            idx: desired index

        Returns:
            postion in meters at the desired frame 
        """
        raise NotImplementedError

    def rotation(self, idx) -> NDArray:
        """Return rotation (yaw, roll, pitch) in radians at position idx

        Args:
            idx: desired index

        Returns:
            rotation in radians at the desired frame
        """
        raise NotImplementedError
    
    def smooth_position(self, idx) -> NDArray:
        """Return smoothed position (x,y,z) in meters at position idx

        Args:
            idx: desired index

        Returns:
            smooth position in meters at the desired frame
        """
        raise NotImplementedError

    def smooth_heading(self, idx) -> float:
        """Return smoothed yaw (vehicle rotation around up axis) in radians 
        at position idx

        Args:
            idx: desired index
        Returns:
            smooth rotation in radians at the desired frame
        """
        raise NotImplementedError

    def restart_from_(self, start_index=0) -> None:
        """Set start position of the dataset

        Args:
            start_index: index of the first frame. Defaults to 0.
        """
        raise NotImplementedError

    def __getitem__(self, idx) -> Tuple[NDArray, NDArray, NDArray]:
        """Return a tuple with image, position and rotation in the specified 
        frame
        """
        return self.image(idx), self.position(idx), self.rotation(idx)

    def __len__(self) -> int:
        raise NotImplementedError