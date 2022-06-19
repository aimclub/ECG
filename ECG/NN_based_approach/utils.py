import torch.nn as nn
import torch
import numpy as np


def shape_change_conv(input_shape, kernel_size, padding, stride):
    out_shape = [0, 0]
    # int((H_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
    numerator_0 = (input_shape[0] + 2 * padding[0] - 1 * (kernel_size[0] - 1) - 1)
    numerator_1 = (input_shape[1] + 2 * padding[1] - 1 * (kernel_size[1] - 1) - 1)
    out_shape[0] = int(numerator_0 / stride[0] + 1)
    out_shape[1] = int(numerator_1 / stride[1] + 1)
    return out_shape


def signal_rescale(signal: np.array, up_slice):
    return np.array([subarray[:up_slice] for subarray in signal])


def make_standard_layer(in_channels, out_channels, kernel_size,
                        padding, stride, LReLU_coef,
                        pool_kernel_size, pool_padding, pool_stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=kernel_size,
                  padding=padding, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(LReLU_coef),
        nn.MaxPool2d(kernel_size=pool_kernel_size,
                     padding=pool_padding, stride=pool_stride)
    )


class DoublePathLayer(nn.Module):
    def __init__(self, layer1, layer2, layerunite, unite='concat'):
        super(DoublePathLayer, self).__init__()
        assert unite in ['sum', 'concat'], 'unite must be "sum" or "concat"'
        self.layer1 = layer1
        self.layer2 = layer2
        self.layerunite = layerunite
        self.unite = unite

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        if self.unite == 'sum':
            x1 = x1 + x2
        elif self.unite == 'concat':
            x1 = torch.cat([x1, x2], dim=1)
        x1 = self.layerunite(x1)
        return x1


def make_standard_double_layer(in_ch=1,
                               out_ch=2,
                               kernel_size=(1, 50),
                               padding=(3, 30),
                               stride=(2, 4),
                               dropout_coef=0.5):
    return DoublePathLayer(
        layer1=nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                      kernel_size=kernel_size,
                      padding=padding, stride=stride),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(dropout_coef),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch,
                      kernel_size=(1, 9), padding=(0, 4), stride=(1, 1))
        ),
        layer2=nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size, padding=padding, stride=stride),
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                      kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
        ),
        layerunite=nn.Sequential(
            nn.BatchNorm2d(out_ch * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(dropout_coef)
        ),
        unite='concat'
    )
