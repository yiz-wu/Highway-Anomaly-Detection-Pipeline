import numpy as np
import cv2

class MergeAdapter:
    """Base class for bev merge functions"""

    def merge(self, main, other):
        """Merge two bev togherer and return the result (should be a 2d integer
        matrix containing the labels)

        Args:
            main: main image to merge
            other: secondary image to merge
        """
        raise NotImplementedError
    
    def result_center(self):
        """Return the center of rotation of the merged image"""
        raise NotImplementedError

    def result_size(self):
        """Return 2d size (no depth) of the result"""
        raise NotImplementedError



class InplaceAddictiveAdapter(MergeAdapter):
    """Adapter that adds other prediction to the main one only where no label 
    is predicted for a pixel
    """
    def __init__(self, size, center) -> None:
        super().__init__()
        self.size = size
        self.center = center

    def merge(self, main, other):
        #TODO: fix
        t = main if main.ndim <= 2 else main.sum(axis=-1)

        return main + other * (main == 0).astype(np.uint8)
    
    def result_center(self):
        return self.center

    def result_size(self):
        return np.asarray(self.size) 


class UpDownAddictiveAdapter(MergeAdapter):
    """Merge two bev, the main up and the other down.
    Note: the bev forward direction must be up
    """
    def __init__(self, size, center) -> None:
        '''
        Args:
            size: size of the resulting image (H,W)
            center: center of the resulting image
        '''
        super().__init__()
        self.size = size
        self.center = center

    def merge(self, main, other):
        if main.ndim > 2:
            res = np.zeros( (*self.size, main.shape[-1]) ) # (H,W,C)
        else:
            res = np.zeros( self.size )
        vertical_separator = int(self.size[0] / 2)
        res[:vertical_separator, ...] = main
        res[vertical_separator: 2* vertical_separator, ...] = cv2.rotate(other, cv2.ROTATE_180)

        return res.astype(np.uint8).squeeze()
        
    def result_center(self):
        return self.center

    def result_size(self):
        return np.asarray(self.size) 

