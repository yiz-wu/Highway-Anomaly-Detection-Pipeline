from .dataloader import DataLoader
from typing import Tuple
from numpy.typing import NDArray

class CompositeDataLoader(DataLoader):
    """
    Class used to compose complex data loading pipelines. 
    This class is designed to be nested with itself as many times as you want.
    At every nest a tuple with (main, other) is returned; main and other can 
    be both a image and a tuple. This mechanism is designed to allow the 
    returned tuples be unrolled by multipredicotrs with the same amount of 
    nesting

    Note: the main dataloder is responsible to return the vehicle position and 
    rotation
    """
    def __init__(self, main: DataLoader, other:DataLoader) -> None:
        super().__init__()
        assert len(main) == len(other), "Incompatible dataloader len"
        self.main = main
        self.other = other

    def image(self, idx) -> NDArray:
        raise (self.main.image(idx), self.other.image(idx))

    def position(self, idx) -> NDArray:
        return self.main.position(idx)

    def rotation(self, idx) -> NDArray:
        return self.main.rotation(idx)
    
    def smooth_position(self, idx) -> NDArray:
        return self.main.rotation(idx)

    def smooth_heading(self, idx) -> float:
        return self.main.smooth_heading(self, idx)

    def restart_from_(self, start_index=0) -> None:
        """Set start position of the dataset"""
        self.main.restart_from_(start_index)
        self.other.restart_from_(start_index)

    def __getitem__(self, idx) -> Tuple[Tuple, NDArray, NDArray]:   
        i_m, p, r = self.main[idx]
        i_o, _, _ = self.other[idx]  
        return (i_m, i_o), p, r

    def __len__(self) -> int:
        return len(self.main)