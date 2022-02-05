import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    'MLP',
    'ConvBlock2D', 'ResBlock2D',
    'ConvBlock3D', 'ResBlock3D'
]



class MLP(nn.Module):
    """
    MLP Model.
    Args:
        input_dim (int) : Dimension of the network input.
        output_dim (int): Dimension of the network output.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.
    Return:
        The output torch.Tensor of the MLP
    """

    def __init__(self, input_dim, output_dim, hidden_sizes, hidden_nonlinearity=F.relu, hidden_w_init=nn.init.xavier_normal_, hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None, output_w_init=nn.init.xavier_normal_, output_b_init=nn.init.zeros_, layer_normalization=False):
        super().__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_nonlinearity = hidden_nonlinearity
        self._output_nonlinearity = output_nonlinearity
        self._layer_normalization = layer_normalization
        self._layers = nn.ModuleList()

        prev_size = input_dim

        for size in hidden_sizes:
            layer = nn.Linear(prev_size, size)
            hidden_w_init(layer.weight)
            hidden_b_init(layer.bias)
            self._layers.append(layer)
            prev_size = size

        layer = nn.Linear(prev_size, output_dim)
        output_w_init(layer.weight)
        output_b_init(layer.bias)
        self._layers.append(layer)

    def forward(self, input_val):
        """Forward method."""
        B = input_val.size(0)
        x = input_val.view(B, -1)
        for layer in self._layers[:-1]:
            x = layer(x)
            if self._hidden_nonlinearity is not None:
                x = self._hidden_nonlinearity(x)
            if self._layer_normalization:
                x = nn.LayerNorm(x.shape[1])(x)

        x = self._layers[-1](x)
        if self._output_nonlinearity is not None:
            x = self._output_nonlinearity(x)

        return x

class ConvBlock2D(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, padding=None, stride=1, dilation=1, norm=False, activation=None, pool=False, upsm=False):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=not norm)
        self.norm = nn.BatchNorm2d(planes) if norm else None
        self.activation = get_activation(activation)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if pool else None
        self.upsm = upsm

    def forward(self, x):
        out = self.conv(x)

        out = out if self.norm is None else self.norm(out)
        out = out if self.activation is None else self.activation(out)
        out = out if self.pool is None else self.pool(out)
        out = out if not self.upsm else F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)

        return out


class ConvBlock3D(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, padding=None, stride=1, dilation=1, norm=False, activation=None, pool=False, upsm=False):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv3d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=not norm)
        self.norm = nn.BatchNorm3d(planes) if norm else None
        self.activation = get_activation(activation)
        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1) if pool else None
        self.upsm = upsm

    def forward(self, x):
        out = self.conv(x)

        out = out if self.norm is None else self.norm(out)
        out = out if self.activation is None else self.activation(out)
        out = out if self.pool is None else self.pool(out)
        out = out if not self.upsm else F.interpolate(out, scale_factor=2, mode='trilinear', align_corners=True)

        return out


class ResBlock2D(nn.Module):
    def __init__(self, inplanes, planes, downsample=None, activation='relu', last_activation=True):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.activation1 = get_activation(activation)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.activation2 = get_activation(activation) if last_activation else None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = out if self.activation1 is None else self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = out if self.activation2 is None else self.activation1(out)

        return out


class ResBlock3D(nn.Module):
    def __init__(self, inplanes, planes, downsample=None, activation='relu', last_activation=True):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.activation1 = get_activation(activation)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.activation2 = get_activation(activation) if last_activation else None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = out if self.activation1 is None else self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = out if self.activation2 is None else self.activation1(out)

        return out


def get_activation(activation):
    if activation==None:
        return None
    elif activation=='relu':
        return nn.ReLU(inplace=True)
    elif activation=='selu':
        return nn.SELU(inplace=True)
    elif activation=='leaky_relu':
        return nn.LeakyReLU(inplace=True)
    else:
        raise NotImplementedError(f'Does not support {activation} activation.')
