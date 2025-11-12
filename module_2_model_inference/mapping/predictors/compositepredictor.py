from .predictor import Predictor, StandardPostProcessPredictor
from numpy.typing import NDArray
from typing import Union
from .mergeadapters import MergeAdapter


class CompositePredictor(Predictor):
    """
    Predictor designed to merge two bev images.
    To merge the prediction you can specify the merge function

    This predcitor can be nested as many time as required to match the 
    dataloader structure
    """
    def __init__(
        self,
        main: Predictor,
        other: Predictor,
        composer: MergeAdapter
        ) -> None:
        """Constructor

        Args:
            main: main predictor
            other: other predictor
            composer: merge adapter used to glue predictor's outputs
        """
        super().__init__()

        assert main.get_map_resolution() == other.get_map_resolution(), \
            "Predictors must have the same map resolution"

        self.main = main
        self.other = other
        self.composer = composer

    def predict(self, img) -> NDArray:
        i_main, i_other = img
        return self.main.predict(i_main), self.other.predict(i_other)
        
    def predict_bev(self, img) -> NDArray:
        i_main, i_other = img
        m = self.main.post_process_bev(self.main.predict_bev(i_main))
        o = self.other.post_process_bev(self.other.predict_bev(i_other))
        return self.composer.merge(m,o)

    def get_bev_to_forward_angle(self) -> float:
        return self.main.get_bev_to_forward_angle()

    def get_bev_center(self) -> Union[tuple, None]:
        return self.composer.result_center()


    def get_map_depth(self) -> float:
        return self.main.get_map_depth()

    def get_bev_size(self) -> NDArray:
        return self.composer.result_size()


    def get_map_resolution(self) -> float:
        return self.main.get_map_resolution()


class StandardCompositePredictor(CompositePredictor, StandardPostProcessPredictor):
    """
    Predictor designed to merge two bev images. This prediction includes 
    standard postprocessing.
    If you dont need post processing use the base CompositePredictor that has 
    no protprocessing.

    This predcitor can be nested as many time as required to match the 
    dataloader structure
    """
    def __init__(
        self,
        main: Predictor,
        other: Predictor,
        composer: MergeAdapter, 
        min_block_area: float = 20,
        drivable_class=None
        ) -> None:
        CompositePredictor.__init__(self, main, other, composer)
        StandardPostProcessPredictor.__init__(self, min_block_area, drivable_class)