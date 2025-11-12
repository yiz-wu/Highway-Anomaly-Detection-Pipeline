"""Package containing RM models based on paper: A Lane-Level Road Marking Map Using a Monocular Camera
"""
import torch
import torch.nn as nn
import torchvision

import torchvision.models.resnet as resnet
from .encoders import get_encoder


def deconv_layer(input_channels, output_channels, stride=1, kernel_size=2, padding="same") -> nn.Sequential:
    """Make a simple deconvolution block used to build the decoder"""
    return nn.Sequential(
        #nn.Upsample(scale_factor=2, mode="nearest"),
        #nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride, padding=padding),
        nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2, padding=0),
        nn.ReLU()
        #nn.SiLU()
    )

def deconv_layer_out(input_channels, output_channels, stride=1, kernel_size=2, padding="same") -> nn.Sequential:
    """Make a simple deconvolution block used to build the decoder"""
    return nn.Sequential(
        #nn.Upsample(scale_factor=2, mode="nearest"),
        #nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride, padding=padding),
        nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2, padding=0),
        nn.ReLU()
    )


class RmConvolutionPipe(nn.Module):
    """
    RmConvolutionPipe is made of tree convolutions layer and is used to connect the backbone to the decoder
    """

    def __init__(self, input_channels, output_channels, kernel_size=2, conv_depth=4096):
        """
        :param input_channels: the number of channels of the first convolution
        :param output_channels: number of output channels. I believe the paper uses the total number of classes
        :param conv_depth: number of channels used in intermediate convolutions
        :param kernel_size: kernel size of all the convolutions
        """
        super().__init__()

        self.pipe = nn.Sequential(
            nn.Conv2d(input_channels, conv_depth, kernel_size, padding="same", bias=False, ),
            nn.SiLU(),
            nn.Conv2d(conv_depth, conv_depth, kernel_size, padding="same", bias=False, ),
            nn.SiLU(),
            nn.Conv2d(conv_depth, output_channels, kernel_size, padding="same", bias=False, ),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.pipe(x)


class RMDecoder(nn.Module):
    """
    RmNetDecoder is the decoder block used in the paper.
    This does not include the "convolution pipe" that connects the backbone to the decoder
    Inputs sizes must be multiple of 32 to work as expected
    """

    def __init__(self, input_channels, output_classes, inner_skip_channels, outer_skip_channels, deconv_depth=64,
                 kernel_size=3):
        super().__init__()

        self.blocks = nn.ModuleList()

        self.blocks.append(
            deconv_layer(input_channels, deconv_depth * 4, stride=1, kernel_size=kernel_size, )
        )
        self.blocks.append(
            deconv_layer(deconv_depth * 4, deconv_depth * 3, kernel_size=kernel_size, stride=1)
        )
        self.blocks.append(
            deconv_layer(deconv_depth * 3 + inner_skip_channels, deconv_depth * 2, kernel_size=kernel_size, stride=1)
        )
        self.blocks.append(
            deconv_layer(deconv_depth * 2 + outer_skip_channels, deconv_depth, kernel_size=kernel_size, stride=1)
        )
        self.blocks.append(
            deconv_layer_out(deconv_depth, output_classes, kernel_size=kernel_size, stride=1)
        )

    def forward(self, x, inner_skip, outer_skip):
        #print("after pipe ", x.shape)
        x = self.blocks[0](x)
        x = self.blocks[1](x)
        #print(x.shape)
        #print(inner_skip.shape)
        x = torch.cat((x, inner_skip), 1)
        x = self.blocks[2](x)
        #print(x.shape)
        #print(outer_skip.shape)
        x = torch.cat((x, outer_skip), 1)
        x = self.blocks[3](x)
        return self.blocks[4](x)


class RmNet(nn.Module):
    """
    RmNet alternative implementation with complete resnet 18 as backbone
    """

    def __init__(self, line_classes: int, mark_classes: int, lock_grad_perc=0, pipe_depth = 2048, decoder_depth = 128):
        super().__init__()

        self.line_classes = line_classes
        self.mark_classes = mark_classes
        self.lock_grad_perc = lock_grad_perc
        total_classes = line_classes + mark_classes

        rn = torchvision.models.resnet18(pretrained=True)
        # keep only conv blocks as separate items, so we can add skip connections later
        self.backbone = nn.ModuleList(list(rn.children())[:-2])
        # fine tune only a part of the backbone
        n = int(self.lock_grad_perc * len(self.backbone))
        if n > 0:
            print(f"Locked {n} backbone layers")
            for i in range(n):
                self.backbone[i].requires_grad_(False)

        self.line_pipe = RmConvolutionPipe(input_channels=512, output_channels=total_classes, conv_depth=pipe_depth)
        self.line_decoder = RMDecoder(total_classes, line_classes, 128, 64, deconv_depth=decoder_depth)

        self.marks_pipe = RmConvolutionPipe(input_channels=512, output_channels=total_classes, conv_depth=pipe_depth)
        self.marks_decoder = RMDecoder(total_classes, mark_classes, 128, 64, deconv_depth=decoder_depth)

        self.single_output = False

    def forward(self, x):
        mid_results = [x]
        for i in range(len(self.backbone)):
            mid_results.append(self.backbone[i](mid_results[i]))

        line = self.line_pipe(mid_results[-1])
        line = self.line_decoder(line, mid_results[6], mid_results[5])

        if not self.single_output:
            marks = self.marks_pipe(mid_results[-1])
            marks = self.marks_decoder(marks, mid_results[6], mid_results[5])

            return line, marks
        else:
            return line

    def use_single_output(self, enable):
        self.single_output = enable


class RmNetE(nn.Module):
    """
    RmNet implementation with variable backbone
    """
    def __init__(self, line_classes: int, mark_classes: int, pipe_depth = 2048, decoder_depth = 128, backbone="resnet-18"):
        super(RmNetE, self).__init__()

        self.line_classes = line_classes
        self.mark_classes = mark_classes
        total_classes = line_classes + mark_classes

        self.backbone = get_encoder(backbone, in_channels=3, depth=5, weights='imagenet')

        # auto-detect depths
        outs = self.backbone(torch.ones((1, 3, 2**6, 2**6))) # fake an input to get depths

        # change with backbone
        pipe_in_channels = outs[-1].shape[1]
        inner_skip_channels = outs[-3].shape[1]
        outer_skip_channels = outs[-4].shape[1]
        print(f"Detected output channels: {pipe_in_channels}")
        print(f"Detected skip connection channels: {inner_skip_channels} -> {outer_skip_channels}")

        self.line_pipe = RmConvolutionPipe(input_channels=pipe_in_channels, output_channels=total_classes, conv_depth=pipe_depth)
        self.line_decoder = RMDecoder(total_classes, line_classes, inner_skip_channels, outer_skip_channels, deconv_depth=decoder_depth)

        self.marks_pipe = RmConvolutionPipe(input_channels=pipe_in_channels, output_channels=total_classes, conv_depth=pipe_depth)
        self.marks_decoder = RMDecoder(total_classes, mark_classes, inner_skip_channels, outer_skip_channels, deconv_depth=decoder_depth)


    def forward(self, x):
        mid_results = self.backbone(x)

        line = self.line_pipe(mid_results[-1])
        line = self.line_decoder(line, mid_results[-3], mid_results[-4])

        marks = self.marks_pipe(mid_results[-1])
        marks = self.marks_decoder(marks, mid_results[-3], mid_results[-4])

        return line, marks

class RmNetPaper(nn.Module):
    """
    RNet implementation as close as possible to the paper: A Lane-Level Road Marking Map Using a Monocular Camera
    """

    def __init__(self, line_classes: int, mark_classes: int, base_depth: int = 64) -> None:
        """
        :param line_classes: number of classes returned as output for line segmentation decoder
        :param base_depth: convolution depth used to scale all the others, 64 is the default scaling to generate the
        networks as close as possible to the paper
        """
        super().__init__()

        self.line_classes = line_classes
        self.base_depth = base_depth
        self.mark_classes = mark_classes

        total_classes = self.line_classes + self.mark_classes

        # first convolution emulate resnet 18
        # we skip the max pooling layer to only reduce the size by two like the paper
        self.in_conv_block = nn.Sequential(
            nn.Conv2d(3, base_depth, kernel_size=7, stride=2, bias=False, padding=3),
            nn.BatchNorm2d(base_depth, affine=True),
            nn.ReLU(),
        )

        # prepare three resnet blocks
        self.res1 = self._make_resnet_layer(base_depth, base_depth * 2, stride=2)  # outer skip
        self.res2 = self._make_resnet_layer(base_depth * 2, base_depth * 4, stride=2)  # inner skip
        self.res3 = self._make_resnet_layer(base_depth * 4, base_depth * 8, stride=2)

        self.line_seg_pipe = RmConvolutionPipe(input_channels=base_depth * 8, output_channels=total_classes,
                                               conv_depth=base_depth * 16)
        self.line_decoder = self._make_decoder(total_classes, base_depth, base_depth * 4, base_depth * 2,
                                               line_classes, )

        self.mark_seg_pipe = RmConvolutionPipe(input_channels=base_depth * 8, output_channels=total_classes,
                                               conv_depth=base_depth * 16)
        self.mark_decoder = self._make_decoder(total_classes, base_depth, base_depth * 4, base_depth * 2,
                                               mark_classes, )

    def _make_resnet_layer(self, input_depth, output_depth, stride=1) -> nn.Sequential:
        """
        Generate a resnet 18 layer
        :param input_depth: number of channels of the input
        :param output_depth: number of channels of the output
        :param stride: stride used by the first convolution
        """
        downsample = nn.Sequential(
            resnet.conv1x1(input_depth, output_depth, stride),
            nn.BatchNorm2d(output_depth),
        )

        return nn.Sequential(
            resnet.BasicBlock(inplanes=input_depth, planes=output_depth, stride=stride, downsample=downsample),
            resnet.BasicBlock(inplanes=output_depth, planes=output_depth)
        )

    def _make_decoder(self, input_channels, base_depth, inner_skip_depth, outer_skip_depth,
                      output_depth) -> nn.ModuleList:
        """
        Create the decoder module list, this returns a non callable block.
        The decoder should be used with _decoder_forward() function
        :param input_channels: input channels of the first deconvolution
        :param base_depth: based depth used to scale the upconvolutions
        :param inner_skip_depth: number channels of the inner skip connection
        :param outer_skip_depth: number of channels of the outer skip connection
        :param output_depth: number of output classes for this decoder
        """
        decoder = nn.ModuleList()
        decoder.append(deconv_layer(input_channels, base_depth * 8))
        decoder.append(deconv_layer(base_depth * 8 + inner_skip_depth, base_depth * 4))
        decoder.append(deconv_layer(base_depth * 4 + outer_skip_depth, base_depth * 2))
        decoder.append(deconv_layer(base_depth * 2, output_depth))
        decoder.append(nn.Softmax(dim=1))

        return decoder

    def forward(self, x):
        # print("original shape: ", x.shape)
        x = self.in_conv_block(x)
        outer_skip = self.res1(x)
        inner_skip = self.res2(outer_skip)
        x = self.res3(inner_skip)

        # print("Encoder shape: ", x.shape)
        # decoder 1
        x1 = self.line_seg_pipe(x)
        # print("Pipe shape: ", x1.shape)
        line_seg = self._decoder_forward(self.line_decoder, x1, inner_skip, outer_skip)
        # print("Decoder shape: ", line_seg.shape)

        x2 = self.mark_seg_pipe(x)
        # print("Pipe shape: ", x1.shape)
        mark_seg = self._decoder_forward(self.mark_decoder, x2, inner_skip, outer_skip)

        return line_seg, mark_seg

    def _decoder_forward(self, decoder, x, inner_skip, outer_skip):
        """
        Make forward pass with a decoder and return the prediction
        :param decoder: decoder to use to calculate the forward pass
        :param x: decoder input
        :param inner_skip: inner skip connection data
        :param outer_skip: outer skip connection data
        :return decoder prediction
        """
        x = decoder[0](x)
        x = torch.cat((x, inner_skip), 1)
        x = decoder[1](x)
        # print(x.shape)
        # print(outer_skip.shape)
        x = torch.cat((x, outer_skip), 1)
        x = decoder[2](x)
        x = decoder[3](x)
        return decoder[4](x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    """Used to see model structure by running this module as main"""
    from torchinfo import summary

    test_input = torch.ones(1, 3, 320, 640)

    model = RmNetPaper(line_classes=7, mark_classes=3).to("cpu")
    # print(model)
    print(f"{model.__class__.__name__}: {count_parameters(model):,} total trainable parameters", )
    line, mark = model(test_input)
    print(line.shape)
    print(mark.shape)

    model = RmNet(line_classes=7, mark_classes=3)
    print(f"{model.__class__.__name__}: {count_parameters(model):,} total trainable parameters", )
    line, mark = model(test_input)
    print(line.shape)
    print(mark.shape)

    # batch_size = 1
    # summary(model, input_size=(batch_size, 3, 720, 1280), device="cpu")

    # emulate a simple batch with a single image just to test out that the output has still the same dimension
