from . import PostProcessor
from typing import List
import networkx as nx
from tqdm import tqdm

class FallbackClassProcessor(PostProcessor):
    """Processor designed to replace prediction classes on graphs. You can use
    this processor in two ways:
        * replace mode: replace an existing class with a desired one
        * fallback mode: replace ad existing class with the fallback class if it
            is present the the graph nodes. If no fallback class attribute can 
            be found nothing happens.

    If you provide settings for both modes, fallback mode has the precedence to
    replace. This means that is a valid fallback class can be replaced, the node
    won't be set to the required replace class.
    """
    def __init__(
        self,
        target_class: int,
        valid_fallback: List[int] = None,
        set_class: int = None,
        consider_boxes = False,
        ) -> None:
        """
        Args:
            target_class: only nodes of this class will be cosidered for class
                changes. If none all classes are used as targets
            valid_fallback: list of valid fallback classes that can be used to 
                replace the current node class
            set_class: new class to set on matching nodes
            consider_boxes: by default graphs are marked as boxes are ignore. 
                Set to true also consider them in class changes. This is not 
                recommended because boxes are usually placed by other 
                postprocessors and contain valid areas of the same class that 
                does not need changes.
        """
        super().__init__()

        assert valid_fallback is not None or set_class is not None, "Either" \
            "valid_fallback or set_class must have a value"
        self.target_class = target_class
        self.valid_fallback = valid_fallback
        self.set_class = set_class
        self.consider_boxes = consider_boxes


    def process(self, graphs: List[nx.Graph]) -> List[nx.Graph]:

        set_nodes_class_num = 0
        fallback_node_class_num = 0

        for g in tqdm(graphs):
            
            # ignore box graphs generated in other steps
            if not self.consider_boxes and "type" in g.graph and g.graph["type"] == "box":
                continue

            for n in g.nodes(data=True):
                current_class = n[1]["attributes"]["predicted_class"] 
                if self.target_class is not None and current_class != self.target_class:
                    continue
                
                ok = False

                # set fallback class if possible
                if self.valid_fallback is not None and "fallback_class" in n[1]["attributes"]:
                    fallback_class = n[1]["attributes"]["fallback_class"] 
                    if fallback_class in self.valid_fallback:
                        n[1]["attributes"]["predicted_class"] = fallback_class
                        fallback_node_class_num += 1
                        ok = True
                
                # set hand picked class
                if not ok and self.set_class is not None:
                    n[1]["attributes"]["predicted_class"] = self.set_class 
                    set_nodes_class_num += 1
        
        print(f"Updated classes: {fallback_node_class_num} fallbacks - {set_nodes_class_num} replaced")

        return graphs


        