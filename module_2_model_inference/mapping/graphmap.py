from .regioniterable import RegionIterable
from .graphs.imageparser import GraphData, ImageParserBase
from .graphs.postprocessing import PostProcessor

from numpy.typing import NDArray
from typing import List
import networkx as nx
from networkx.readwrite import json_graph
import os
from tqdm import trange
import shutil
import json


class GraphMap:
    """Class used to generate graph maps.

    At the moment the main purpose of this class is to be a sort of wrapper to
    easly transform images into graphs and saving them into standard formats
    that can be processed in other tools.

    This class uses a internal format to hold graphs that cannot saved easly on 
    disk. For this reason this class does not provile loading functionalities.
    """

    def __init__(
        self,
        parser: ImageParserBase,       
    ) -> None:
        """

        Args:
            parser: parser used to generate graphs
        """
        self.parser = parser
        self.data: GraphData = None

        self.nx_graphs_data = None
        self.did_postprocess = False
    
    def parse_image(
        self,
        img: NDArray,
        center_anchor: NDArray,
        rotation_center: List[int],
        rotation_to_vertical: float,
        rotation_to_world: float,
        ):
        """Manually add a single image to the current map

        Args:
            img: image to parse
            center_anchor: center of rotation coordinates in world
            rotation_center: center of rotation in pixels for this image
            rotation_to_vertical: rotation applied to make the image have most of
                the lines with vertical orientation
            rotation_to_world: rotation applied to bring back the extracted point
                to their correct world positions
        """
        if self.did_postprocess:
            msg = "Cannot add data to a postprocessed map." \
            " Postprocessing invalidates internal data and caches so no new" \
            " data can be added"
            raise Exception(msg)

        self.data = self.parser.parse(
            img,
            center_anchor,
            rotation_center,
            rotation_to_vertical,
            rotation_to_world,
            data=self.data
        )


    def from_iterable(
        self,
        iterable: RegionIterable,
        start_frame: int = 0,
        end_frame: int = -1,
        step: int = 1 
        ):
        """Generate graph map from iterable. The iterable should return regions
        ordered in a way that is supported by the parser

        Args:
            iterable: region iterable used to iterate over all the map regions and
                generate the graph output.
            start_frame: start index of the iterable. Defaults to 0.
            end_frame: end index of the iterable. Pass -1 for last index available.
                Defaults to -1.
            step: interation step to jump regions. Defaults to 1.
        """
        if end_frame == -1:
            end_frame = len(iterable)

        assert start_frame < end_frame, "end frame must be greater then start frame"
        
        assert step >= 1, "Step should be greater than 0"

        for f in trange(start_frame, end_frame, step):
            self.parse_image(*iterable[f])

    def num_graphs(self, min_graph_nodes = 2) -> int:
        """Return the number of graphs in this map"""
        graphs = self.nx_graphs(min_graph_nodes)
        return len(graphs)

    def num_nodes(self, min_graph_nodes = 2) -> int:
        """Return the number of nodes in this map"""
        graphs = self.nx_graphs(min_graph_nodes)
        
        sz = 0
        for g in graphs:
            sz += len(g.nodes)
        
        return sz

    def nx_graphs(self, min_graph_nodes = 2) -> List[nx.Graph]:
        """Convert internal graphs into networkx graphs

        Args:
            min_graph_nodes: minimum nodes for a graph to be valid. Defaults to 2.

        Returns:
            list of networkx graphs with available attributes
            attributes: {"position": NDArray, }
        """

        assert min_graph_nodes >= 2, "min_graph_nodes must be >= 2"

        if self.nx_graphs_data is None:
            self.nx_graphs_data = self.__create_nx_graphs()

        graphs = [g for g in self.nx_graphs_data if g.number_of_nodes() >= min_graph_nodes]    
        
        return graphs
    

    def __create_nx_graphs(self):
        graphs = []
        for g in self.data.graphs:
            # ignore single point graphs that are not graphs
            if g.number_of_nodes() < 2:
                continue

            # prepare attributes
            centers = {
                i : {
                        "position": g.centers[i, :],
                        "attributes": g.node_attributes(i),
                    } for i in range(g.number_of_nodes())
            }

            nxg: nx.Graph = nx.from_scipy_sparse_matrix(g.adjacency)
            nx.set_node_attributes(nxg, centers)
            graphs.append(nxg)
        
        return graphs


    def export(self, path: str, min_graph_nodes = 2):
        """Export graphs and attributes to disk in json format

        This function will erase folder at path

        Args:
            path: folder path where the graphs will be saved
            min_graph_nodes: minimum nodes for a graph to be valid. Defaults to 2.
        """
        if os.path.exists(path):
            shutil.rmtree(path)
        
        os.makedirs(path, exist_ok=True)
        graphs = self.nx_graphs(min_graph_nodes)

        for i in range(len(graphs)):
            g = graphs[i]

            save_path = os.path.join(path, f"{i}.json")
            
            with open(save_path, "w") as fp:
                dt = json_graph.node_link_data(g)
                json.dump(dt, fp, cls=NpEncoder)

    def load(self, path: str):
        """Load graphs from json files located in the specified path

        Args:
            path: path containing graph jsons
        """
        self.nx_graphs_data = []
        # lock ability to add more data to this map
        self.did_postprocess = True 
        # read graphs from disk
        for f in os.listdir(path):
            file = os.path.join(path, f)
            with open(file) as fp:
                js_graph = json.load(fp)
                g = json_graph.node_link_graph(js_graph)
                # load only valid graphs
                if g.number_of_nodes() < 2:
                    continue

                # check if graphs has partitions
                # if so split them into separate graphs
                if nx.is_connected(g):
                    self.nx_graphs_data.append(g)
                else:
                    subg = [g.subgraph(c).copy() for c in nx.connected_components(g)]
                    self.nx_graphs_data += subg
                    
        
        print(f"Loaded {len(self.nx_graphs_data)} graphs")

    def postprocess(self, processors: List[PostProcessor]):
        """Execute a list of post processing steps. This function cn be called
        as much as you want. 
        
        Postprocessors are executed sequentially, at every postprocess step the
        processor will receive the result of the previous one.

        The final result of postprocessing will be saved internally and returned
        in nx_graph() and export() functions.

        Args:
            processors: list of post processors to execute
        """
        self.did_postprocess = True

        for i, p in zip(range(len(processors)), processors):

            print(f"{i+1}/{len(processors)} - {p.__class__.__name__}")

            self.nx_graphs_data = p.process(self.nx_graphs())
       

################################################################################
import numpy as np
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "serialize"):
            return obj.serialize()
        return super(NpEncoder, self).default(obj)