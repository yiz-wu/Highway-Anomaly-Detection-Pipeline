"""This package contains dataloader used to load data for the mapping pipeline.

These are not torch dataloader, they are not designed to train models. Their are
designed to provide an easy way to extend and use the mapping pipeline on
different datasets with different formats and requirements without any modification
of the core components of the pipelile.

This dataloaders are also designed to be nested together to compose really complex
data stractures or experiments while keeping the code as simple as possible.
"""
from .dataloader import DataLoader
# from .adegliacco import AdegliaccoLoader, Adegliacco360Loader
from .compositedataloader import CompositeDataLoader
from .sdf import SDFLoader