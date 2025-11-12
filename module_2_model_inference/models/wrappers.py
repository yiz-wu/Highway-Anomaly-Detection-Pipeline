
import torch.nn as nn
import torch

class SingleClassLaneWrapper(nn.Module):
    """Wrapper for networks with two decoders that keep only lines and drops classes
    that are in the ingnore list.
    This returns a 0/1 prediction where 1 are non ignored classes
    """
    def __init__(self, model, ignore_classes = []) -> None:
        super(SingleClassLaneWrapper, self).__init__()

        self.model = model
        self.ignore_classes = ignore_classes


    def forward(self, x):

        line, _= self.model(x)

        line = torch.argmax(line, dim=1)
        for c in self.ignore_classes:
            line[line == c] = 0
        
        line[line != 0] = 1

        return line

 