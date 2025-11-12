# Warped models allow to define a set of custom model outside this repository that
# should work within pipeline
# To create a warped model simply pull the model repository in this folder and then
# create a class in the warped folder that loads the model and allows this pipeline
# to use it
# an output for both drivable and line segmentation
# Note:
#  - every warped model should stay in its directory inside this warped folder
#  - every model should have a "warped" that loades and uses the model 
#  - outputs should have 0 as background class 
#  - drivable prediction have be a 3d tensor with depth equal to two
#
# Rembember to override load_state_dict() into your warp class! This is required
# to actually load the weight into the wrapper model

# Example structure:
# - warped
#       - __init__.py
#       - mywarpedmodel.py // model warp loader
#       - mymodelfolder/
#               - ...

from os.path import dirname
from pkgutil import iter_modules
from importlib import import_module
import sys
import inspect

current_module = sys.modules[__name__]

# automatically import all files in this module
__all__ = [name for _, name, _ in iter_modules([dirname(__file__)])]

for module_name in __all__:  

    module = import_module(__name__ + "." + module_name)
    for c in inspect.getmembers(module, inspect.isclass):
        current_module.__dict__[c[0]] = c[1]

    if len(__all__) == 1:
        globals().update(module.__dict__)

    #print(f"Importing {module.__dict__}")
    del module