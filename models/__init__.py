"""Package containing available torch models

Its recommenended to use RoadStartNetE that is an updated version of RoadStarNet
and allows to use all timm encoders without issues
"""
from .RoadStarNet import RoadStarNet, RoadStarNetE

# import warped classes into this package
import sys, os, importlib.util, inspect, torch.nn as nn

from . import warped as wrp

__current_module = sys.modules[__name__]

for __c in inspect.getmembers(wrp, inspect.isclass):
    __current_module.__dict__[__c[0]] = __c[1]


def load_custom_models(path="custom_models"):
    models_dict = {}
    if not os.path.exists(path):
        return models_dict

    # Ensure parent directory of custom_models is on sys.path
    parent_dir = os.path.abspath(os.path.dirname(path))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    package_name = os.path.basename(path)

    for fname in os.listdir(path):
        # print(f"Checking file: {fname}")

        if fname.endswith(".py") and fname != "__init__.py":
            # print(f"Loading custom model: {fname}")
            module_name = fname[:-3]
            full_name = f"{package_name}.{module_name}"   # e.g. custom_models.HybridDet
            file_path = os.path.join(path, fname)

            spec = importlib.util.spec_from_file_location(full_name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            spec.loader.exec_module(module)

            # Collect all nn.Module subclasses
            for name, cls in inspect.getmembers(module, inspect.isclass):
                if issubclass(cls, nn.Module) and cls is not nn.Module:
                    models_dict[name] = cls
                    print(f"imported {name} : {cls}")

    return models_dict
    