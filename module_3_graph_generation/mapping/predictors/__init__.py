"""Package contaning predictors for mapping pipeline.

A predictor is a wrapper class that has a standard interface that allows to 
define how images received from a dataloder are used to generate a bev for a 
specific frame.

The main purpose of this class is to support any dataset and model or other 
specific user needs. Predictors are a easy and clean way to exten the 
mapping pipeline.

Predictors can be used to change the model or the code that generates the bevs
or reuse the whole mapping pipeline to generate an RGB map of the dataset.
"""

from .predictor import Predictor, StandardPostProcessPredictor
from .modelpredictor import ModelPredictor, SoftmaxPredictor
from .rgbpredictor import RGBPredictor
from .compositepredictor import CompositePredictor, StandardCompositePredictor

# from .adegliacco import Adegliacco360Predictor, AdegliaccoRGB360Predictor

# from .sdf import SDFPredictor