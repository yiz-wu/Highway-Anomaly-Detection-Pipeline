import torch
import torch.nn as nn
import torchvision.models.segmentation as segmentation


class TorchvisionModel(nn.Module):
    def __init__(self, pretrained=True, num_classes=21, checkpoint_path=None, map_location="cpu"):
        super().__init__()
        self.model = segmentation.deeplabv3_resnet50(pretrained=False, pretrained_backbone=False)
        if pretrained and checkpoint_path:
            state = torch.load(checkpoint_path, map_location=map_location)
            self.model.load_state_dict(state, strict=False)
        # Replace classifier if you want custom classes
        if num_classes != 21:
            in_channels = self.model.classifier[-1].in_channels
            self.model.classifier[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)["out"]
