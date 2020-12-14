import torch
from torch import nn


class IdentityLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ConvLayer3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, act_func='relu'):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )
        if act_func == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = None

    def forward(self, x):
        if self.act is None:
            return self.conv(x)
        else:
            return self.act(self.conv(x))


class ManyfoldConvBlock3D(nn.Module):

    def __init__(self, layers, shortcut, skip_pos):
        super(ManyfoldConvBlock3D, self).__init__()
        self.skip_pos = skip_pos
        self.layers = nn.ModuleList(layers)
        self.shortcut = shortcut

    def forward(self, x, skip_x=None):
        if skip_x is None:  # encoder
            skip_x = self.shortcut(x)
        else: # decoder
            skip_x = self.shortcut(skip_x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == self.skip_pos:
                x = x + skip_x
        return x


class TumorSurrogate(nn.Module):
    def __init__(self, widths, n_cells, strides):
        super().__init__()
        input_channel = 128
        first_conv = ConvLayer3D(
            3, input_channel, kernel_size=3, stride=1,
        )

        encoder_blocks = [first_conv]
        for width, n_cell, s in zip(widths, n_cells, strides):
            conv_layers = []
            shortcut = IdentityLayer()
            if s == 1:
                skip_pos = n_cell - 1
            else:
                skip_pos = n_cell - 2
            for i in range(n_cell):
                if i == n_cell - 1:  # last layer of block is pooling or stride conv
                    stride = s
                else:
                    stride = 1
                conv_op = ConvLayer3D(in_channels=input_channel, out_channels=width, kernel_size=3, stride=stride)
                conv_layers.append(conv_op)
                input_channel = width

            conv_block = ManyfoldConvBlock3D(conv_layers, shortcut, skip_pos=skip_pos)
            encoder_blocks.append(conv_block)

        mid_conv = ConvLayer3D(
            input_channel, input_channel - 3, kernel_size=3, stride=1
        )
        encoder_blocks.append(mid_conv)

        decoder_blocks = []
        n_cells_decoder = [x + 1 for x in n_cells]
        for width, n_cell, s in zip(widths, n_cells_decoder, strides):
            conv_layers = []
            if s == 1:
                skip_pos = n_cell - 1
            else:
                skip_pos = n_cell - 2
            shortcut = IdentityLayer()
            for i in range(n_cell):
                if i == n_cell - 1 and s != 1:  # last layer of block is Upsampling
                    conv_op = nn.Upsample(scale_factor=s, mode='nearest')
                else:
                    conv_op = ConvLayer3D(in_channels=input_channel, out_channels=width, kernel_size=3, stride=1)

                conv_layers.append(conv_op)
                input_channel = width
            conv_block = ManyfoldConvBlock3D(conv_layers, shortcut, skip_pos=skip_pos)
            decoder_blocks.append(conv_block)
            if s != 1:
                after_upscale_conv = ConvLayer3D(
                    in_channels=input_channel, out_channels=width,
                    kernel_size=3, stride=1
                )
                decoder_blocks.append(after_upscale_conv)
        # final layer
        last_channel = 1
        last_conv = ConvLayer3D(
            input_channel, last_channel, kernel_size=3, stride=1, act_func = None
        )
        decoder_blocks.append(last_conv)

        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.parameter_encoder = nn.Linear(in_features=3, out_features=3*8*8*8)

    def forward(self, x, parameters):
        for block in self.encoder_blocks:
            x = block(x)
        parameters = self.parameter_encoder(parameters).view(-1, 8, 8, 8, 3).permute(0, 4, 1, 2, 3)

        x = torch.cat((parameters, x), dim=1)
        skip_x = x
        for idx, block in enumerate(self.decoder_blocks):
            if isinstance(block, ManyfoldConvBlock3D):
                x = block(x, skip_x=skip_x)
            else:
                skip_x = x
                x = block(x)
        return x



