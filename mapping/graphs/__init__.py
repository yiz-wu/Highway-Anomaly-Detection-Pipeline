from enum import Enum

class NodeConnectionMode(Enum):
    """Connection mode used to attach nodes together when building the graph map
    """
    SINGLE = "single"
    """Allow a single connection to past nodes
    """
    MULTIPLE = "multiple"
    """Connect to all possible past nodes
    """

from .linegraph import LineGraph

from .imageparser import ImageParserBase
from .boxparser import BoxImageParser