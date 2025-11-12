"""Package containing RoadStartNet models derived from HybridNets"""

import torch
import torch.nn as nn
from .HybridDet import BiFPN, MaxPool2dStaticSamePadding, Conv2dStaticSamePadding
from .encoders import get_encoder


def UpConv(
    input_channels, output_channels, stride=1, kernel_size=3, padding="same"
) -> nn.Sequential:
    """Make a simple deconvolution block used to build the decoder"""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(
            input_channels, output_channels, kernel_size, stride=stride, padding=padding
        ),
        nn.SiLU(),
    )


def UpConvOut(
    input_channels, output_channels, stride=1, kernel_size=3, padding="same"
) -> nn.Sequential:
    """Make a simple deconvolution block used to build the decoder"""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(
            input_channels, output_channels, kernel_size, stride=stride, padding=padding
        ),
    )


def BatchedUpConv(
    input_channels, output_channels, stride=1, kernel_size=3, padding="same"
) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(
            input_channels, output_channels, kernel_size, stride=stride, padding=padding
        ),
        nn.SiLU(),
        nn.BatchNorm2d(output_channels),
    )


class RoadStarDecoder(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        skip_little_channels,
        skip_big_channels,
        top_depth=16,
    ):
        """RoadStartNet Decoder

        Args:
            input_channels: number of channels received as input
            output_channels: number of output channels
            skip_little_channels: depth of innermost skip connection
            skip_big_channels: depth of outer skip connection
            top_depth: base decoder depth used to generate sub-blocks.
                Defaults to 16.
        """
        super(RoadStarDecoder, self).__init__()
        self.output_channels = output_channels
        # all p3 - p7 items are supposed to have the same amount of channels
        self.input_channels = input_channels

        self.p7_to_p6 = UpConv(input_channels, 7 * top_depth)
        self.p6_to_p5 = BatchedUpConv(input_channels + 7 * top_depth, 6 * top_depth)
        self.p5_to_p4 = UpConv(input_channels + 6 * top_depth, 5 * top_depth)
        # print(f"i:{input_channels} +s:{skip_little_channels} + x:{5 * top_depth}")
        self.p4_to_p3 = BatchedUpConv(
            input_channels + skip_little_channels + 5 * top_depth, 4 * top_depth
        )
        self.p3_to_p2 = UpConv(input_channels + 4 * top_depth, 3 * top_depth)
        self.up1 = BatchedUpConv(skip_big_channels + 3 * top_depth, 2 * top_depth)
        self.up2 = UpConvOut(2 * top_depth, output_channels)

    def forward(self, p3_7features, skip_little, skip_big):
        p3, p4, p5, p6, p7 = p3_7features
        x = self.p7_to_p6(p7)
        x = torch.concat((x, p6), dim=1)
        x = self.p6_to_p5(x)
        x = torch.concat((x, p5), dim=1)
        x = self.p5_to_p4(x)
        # print(f"x:{x.shape} - i:{p4.shape} - s:{skip_little.shape}")
        x = torch.concat((x, p4, skip_little), dim=1)
        x = self.p4_to_p3(x)
        x = torch.concat((x, p3), dim=1)
        x = self.p3_to_p2(x)
        x = torch.concat((x, skip_big), dim=1)
        x = self.up1(x)
        return self.up2(x)


class RoadStarNet(nn.Module):
    def __init__(
        self,
        line_classes,
        area_classes,
        compound_coef=3,
        decoder_depth=128,
        skip_area=False,
    ):
        """Inizialize RoadStarNet

        Args:
            line_classes: number of line classes (include background)
            area_classes: number of area/drivable classes (include background)
            compound_coef: EfficientNet version from 1 yto 7. Defaults to 3.
            decoder_depth: Base depth used to generate the network decoder.
                Defaults to 128.
            skip_area: skip drivable area prediction and return (line, none)
        """
        super(RoadStarNet, self).__init__()
        self.compound_coef = compound_coef

        self.skip_area = skip_area

        self.line_classes = line_classes
        self.area_classes = area_classes

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }

        self.skip_little = [112, 120, 136, 160, 176, 200, 200, 200, 200]
        self.skip_big = [24, 24, 32, 32, 40, 40, 40, 40, 40]

        # EfficientNet_Pytorch
        self.encoder = get_encoder(
            "efficientnet-b" + str(self.backbone_compound_coef[compound_coef]),
            in_channels=3,
            depth=5,
            weights="imagenet",
        )

        # BiFPN sequential block
        self.bifpn = nn.Sequential(
            *[
                BiFPN(
                    self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False,
                    use_p8=compound_coef > 7,
                )
                for _ in range(self.fpn_cell_repeats[compound_coef])
            ]
        )

        self.initialize_decoder(self.bifpn)

        self.line_decoder = RoadStarDecoder(
            self.fpn_num_filters[compound_coef],
            line_classes,
            self.skip_little[compound_coef - 1],
            self.skip_big[compound_coef - 1],
            decoder_depth,
        )

        self.area_decoder = RoadStarDecoder(
            self.fpn_num_filters[compound_coef],
            area_classes,
            self.skip_little[compound_coef - 1],
            self.skip_big[compound_coef - 1],
            decoder_depth,
        )

    def forward(self, inputs):
        # p1, p2, p3, p4, p5 = self.backbone_net(inputs)
        encoded = self.encoder(inputs)
        p3, p4, p5 = encoded[-3:]  # self.backbone_net(inputs)

        # print(f"Skip sizes: {encoded[4].shape} - {encoded[2].shape}")

        features = (p3, p4, p5)

        features = self.bifpn(features)

        p3, p4, p5, p6, p7 = features
        # print(f"Features shapes: {p3.shape} - {p4.shape} - {p5.shape} - {p6.shape} - {p7.shape}")

        line_seg = self.line_decoder(features, encoded[4], encoded[2])

        if self.skip_area:
            return line_seg, None

        area_seg = self.area_decoder(features, encoded[4], encoded[2])

        return line_seg, area_seg

    def initialize_decoder(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def freeze_encoder_(self, disable_gradient: bool):
        self.encoder.requires_grad_(not disable_gradient)

    def freeze_decoder_(self, disable_gradient: bool):
        self.bifpn.requires_grad_(not disable_gradient)
        self.area_decoder.requires_grad_(not disable_gradient)
        self.line_decoder.requires_grad_(not disable_gradient)


##############################################################################################################################


class RoadStarNetE(nn.Module):
    """
    Evolution of RoadStarNet that uses custom BiFPN, interchangable backbone
    and custom decoder
    """

    def __init__(
        self,
        line_classes: int,
        area_classes: int,
        decoder_depth: int = 64,
        backbone: str = "timm-tf_efficientnet_lite4",
        num_bifpn: int = 6,
        bifpn_depth: int = None,
        bifpn_activation: bool = True,
        attention: bool = True,
        use_p8: bool = False,
        single_decoder: bool = False,
    ):
        """
        Initialize RoadStarNetE

        Args:
            line_classes: number of classes for line decoder
            area_classes: number of classes for drivable area decoder
            decoder_depth: base depth used to topmost (excluding output)
                convolution of the decoder. This depth is duplicated every step
                down to lower levels of the decoder
            backbone: backbone to use. Reder to timm supported backbones or look
                into models/encoder folder
            num_bifpn: number of bifpn to pipe between encoder and decoder
            bifpn_depth: depth of bifpn convolutions
            bifpn_activation: set to true to enable silu activation after bifpn
                convolutions
            attention: set to true to enable bifpn attenction
                (reccommend to leave true)
            use_p8: should be set to true only for efficientnet-b7 backbone
        """
        super(RoadStarNetE, self).__init__()

        self.line_classes = line_classes
        self.area_classes = area_classes

        # EfficientNet_Pytorch
        self.encoder = get_encoder(
            backbone,
            in_channels=3,
            depth=5,
            weights="imagenet",
        )

        # TODO: check if this can be done statically
        # auto-detect depths using a fake an input to get depths
        outs = self.encoder(torch.ones((1, 3, 2**6, 2**6)))

        # change with backbone
        pipe_in_channels = outs[-1].shape[1]
        inner_skip_channels = outs[4].shape[1]
        outer_skip_channels = outs[2].shape[1]
        print(f"Detected output channels: {pipe_in_channels}")
        print(
            "Detected skip connection channels: "
            f"{inner_skip_channels} -> {outer_skip_channels}"
        )

        p3, p4, p5 = outs[-3:]
        # print(p3.shape[1], p4.shape[1], p5.shape[1])
        bifpn_conv_channels = [p3.shape[1], p4.shape[1], p5.shape[1]]

        bifpn_depth = 160 if bifpn_depth is None else bifpn_depth

        # BiFPN sequential block
        self.bifpn = nn.Sequential(
            *[
                BiFPNE(
                    bifpn_depth,
                    bifpn_conv_channels,
                    out_channels=bifpn_depth,
                    first_time=i == 0,
                    attention=attention,
                    activation=bifpn_activation,
                    use_p8=use_p8,
                )
                for i in range(num_bifpn)
            ]
        )

        self.initialize_decoder(self.bifpn)

        self.line_decoder = RoadDecoderE(
            pipe_in_channels if bifpn_depth is None else bifpn_depth,
            line_classes,
            inner_skip_channels,
            outer_skip_channels,
            decoder_depth,
        )

        self.single_decoder = single_decoder

        if not self.single_decoder:
            self.area_decoder = RoadDecoderE(
                pipe_in_channels if bifpn_depth is None else bifpn_depth,
                area_classes,
                inner_skip_channels,
                outer_skip_channels,
                decoder_depth,
            )

    def forward(self, inputs):
        # p1, p2, p3, p4, p5 = self.backbone_net(inputs)
        encoded = self.encoder(inputs)
        p3, p4, p5 = encoded[-3:]  # self.backbone_net(inputs)

        # print(f"Skip sizes: {encoded[4].shape} - {encoded[2].shape}")

        features = (p3, p4, p5)

        features = self.bifpn(features)

        p3, p4, p5, p6, p7 = features
        # print(f"Features shapes: {p3.shape} - {p4.shape} - {p5.shape} - {p6.shape} - {p7.shape}")

        line_seg = self.line_decoder(features, encoded[4], encoded[2])

        if self.single_decoder:
            return line_seg

        area_seg = self.area_decoder(features, encoded[4], encoded[2])

        return line_seg, area_seg

    def initialize_decoder(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def freeze_encoder_(self, disable_gradient: bool):
        self.encoder.requires_grad_(not disable_gradient)

    def freeze_decoder_(self, disable_gradient: bool):
        self.bifpn.requires_grad_(not disable_gradient)

        if not self.single_decoder:
            self.area_decoder.requires_grad_(not disable_gradient)

        self.line_decoder.requires_grad_(not disable_gradient)


class ConvBlockE(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False):
        super(ConvBlockE, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.norm = norm
        self.activation = activation

        self.depthwise_conv = Conv2dStaticSamePadding(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise_conv = Conv2dStaticSamePadding(
            in_channels, out_channels, kernel_size=1, stride=1
        )

        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's,
            # momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)
        if activation:
            self.act = nn.SiLU()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.act(x)

        return x


class BiFPNE(nn.Module):
    def __init__(
        self,
        num_channels,
        conv_channels,
        first_time=False,
        epsilon=1e-4,
        activation=True,
        attention=True,
        use_p8=False,
        out_channels=None,
    ):
        """

        Args:
            num_channels:
            conv_channels:
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
        """
        super(BiFPNE, self).__init__()
        self.epsilon = epsilon
        self.use_p8 = use_p8

        # Conv layers
        self.conv6_up = ConvBlockE(num_channels, out_channels, activation=activation)
        self.conv5_up = ConvBlockE(num_channels, out_channels, activation=activation)
        self.conv4_up = ConvBlockE(num_channels, out_channels, activation=activation)
        self.conv3_up = ConvBlockE(num_channels, out_channels, activation=activation)
        self.conv4_down = ConvBlockE(num_channels, out_channels, activation=activation)
        self.conv5_down = ConvBlockE(num_channels, out_channels, activation=activation)
        self.conv6_down = ConvBlockE(num_channels, out_channels, activation=activation)
        self.conv7_down = ConvBlockE(num_channels, out_channels, activation=activation)
        if use_p8:
            self.conv7_up = ConvBlockE(
                num_channels, out_channels, activation=activation
            )
            self.conv8_down = ConvBlockE(
                num_channels, out_channels, activation=activation
            )

        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.p5_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.p4_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.p3_upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)
        if use_p8:
            self.p7_upsample = nn.Upsample(scale_factor=2, mode="nearest")
            self.p8_downsample = MaxPool2dStaticSamePadding(3, 2)

        self.swish = nn.SiLU()

        self.first_time = first_time
        if self.first_time:
            self.p5_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            self.p5_to_p6 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                MaxPool2dStaticSamePadding(3, 2),
            )
            self.p6_to_p7 = nn.Sequential(MaxPool2dStaticSamePadding(3, 2))
            if use_p8:
                self.p7_to_p8 = nn.Sequential(MaxPool2dStaticSamePadding(3, 2))

            self.p4_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

        # Weight
        self.p6_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True
        )
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True
        )
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True
        )
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True
        )
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(
            torch.ones(3, dtype=torch.float32), requires_grad=True
        )
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(
            torch.ones(3, dtype=torch.float32), requires_grad=True
        )
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(
            torch.ones(3, dtype=torch.float32), requires_grad=True
        )
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True
        )
        self.p7_w2_relu = nn.ReLU()

        self.attention = attention

    def forward(self, inputs):
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """

        # downsample channels using same-padding conv2d to target phase's if not the same
        # judge: same phase as target,
        # if same, pass;
        # elif earlier phase, downsample to target phase's by pooling
        # elif later phase, upsample to target phase's by nearest interpolation

        if self.attention:
            outs = self._forward_fast_attention(inputs)
        else:
            outs = self._forward(inputs)

        return outs

    def _forward_fast_attention(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2

        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(
            self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in))
        )
        # Weights for P5_0 and P6_1 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(
            self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up))
        )

        # Weights for P4_0 and P5_1 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(
            self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up))
        )

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(
            self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up))
        )

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(
                weight[0] * p4_in
                + weight[1] * p4_up
                + weight[2] * self.p4_downsample(p3_out)
            )
        )

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(
                weight[0] * p5_in
                + weight[1] * p5_up
                + weight[2] * self.p5_downsample(p4_out)
            )
        )

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(
                weight[0] * p6_in
                + weight[1] * p6_up
                + weight[2] * self.p6_downsample(p5_out)
            )
        )

        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(
            self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out))
        )

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)
            if self.use_p8:
                p8_in = self.p7_to_p8(p7_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            if self.use_p8:
                # P3_0, P4_0, P5_0, P6_0, P7_0 and P8_0
                p3_in, p4_in, p5_in, p6_in, p7_in, p8_in = inputs
            else:
                # P3_0, P4_0, P5_0, P6_0 and P7_0
                p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        if self.use_p8:
            # P8_0 to P8_2

            # Connections for P7_0 and P8_0 to P7_1 respectively
            p7_up = self.conv7_up(self.swish(p7_in + self.p7_upsample(p8_in)))

            # Connections for P6_0 and P7_0 to P6_1 respectively
            p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_up)))
        else:
            # P7_0 to P7_2

            # Connections for P6_0 and P7_0 to P6_1 respectively
            p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_up)))

        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_up)))

        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(self.swish(p4_in + p4_up + self.p4_downsample(p3_out)))

        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(self.swish(p5_in + p5_up + self.p5_downsample(p4_out)))

        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(self.swish(p6_in + p6_up + self.p6_downsample(p5_out)))

        if self.use_p8:
            # Connections for P7_0, P7_1 and P6_2 to P7_2 respectively
            p7_out = self.conv7_down(
                self.swish(p7_in + p7_up + self.p7_downsample(p6_out))
            )

            # Connections for P8_0 and P7_2 to P8_2
            p8_out = self.conv8_down(self.swish(p8_in + self.p8_downsample(p7_out)))

            return p3_out, p4_out, p5_out, p6_out, p7_out, p8_out
        else:
            # Connections for P7_0 and P6_2 to P7_2
            p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

            return p3_out, p4_out, p5_out, p6_out, p7_out


class RoadDecoderE(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        skip_little_channels,
        skip_big_channels,
        top_depth=16,
    ):
        super(RoadDecoderE, self).__init__()
        self.output_channels = output_channels
        # all p3 - p7 items are supposed to have the same amount of channels
        self.input_channels = input_channels

        self.p7_to_p6 = RoadDecoderE.upconv(input_channels, 7 * top_depth)
        self.p6_to_p5 = RoadDecoderE.batchedUpconv(
            input_channels + 7 * top_depth, 6 * top_depth
        )
        self.p5_to_p4 = RoadDecoderE.upconv(
            input_channels + 6 * top_depth, 5 * top_depth
        )
        # print(f"i:{input_channels} +s:{skip_little_channels} + x:{5 * top_depth}")
        self.p4_to_p3 = RoadDecoderE.batchedUpconv(
            input_channels + skip_little_channels + 5 * top_depth, 4 * top_depth
        )
        self.p3_to_p2 = RoadDecoderE.upconv(
            input_channels + 4 * top_depth, 3 * top_depth
        )
        self.up1 = RoadDecoderE.batchedUpconv(
            skip_big_channels + 3 * top_depth, 2 * top_depth
        )
        self.up2 = RoadDecoderE.upconvOut(2 * top_depth, output_channels)

    def forward(self, p3_7features, skip_little, skip_big):
        p3, p4, p5, p6, p7 = p3_7features
        x = self.p7_to_p6(p7)
        x = torch.concat((x, p6), dim=1)
        x = self.p6_to_p5(x)
        x = torch.concat((x, p5), dim=1)
        x = self.p5_to_p4(x)
        # print(f"x:{x.shape} - i:{p4.shape} - s:{skip_little.shape}")
        x = torch.concat((x, p4, skip_little), dim=1)
        x = self.p4_to_p3(x)
        x = torch.concat((x, p3), dim=1)
        x = self.p3_to_p2(x)
        x = torch.concat((x, skip_big), dim=1)
        x = self.up1(x)
        return self.up2(x)

    @staticmethod
    def upconv(
        input_channels, output_channels, stride=1, kernel_size=3, padding="same"
    ) -> nn.Sequential:
        """Make a simple deconvolution block used to build the decoder"""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.SiLU(),
        )

    @staticmethod
    def upconvOut(
        input_channels, output_channels, stride=1, kernel_size=3, padding="same"
    ) -> nn.Sequential:
        """Make a simple deconvolution block used to build the decoder"""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                stride=stride,
                padding=padding,
            ),
        )

    @staticmethod
    def batchedUpconv(
        input_channels, output_channels, stride=1, kernel_size=3, padding="same"
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.SiLU(),
            nn.BatchNorm2d(output_channels),
        )
