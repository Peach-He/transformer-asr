import math
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Tuple

from model.module.containers import Sequential
from model.module.normalization import LayerNorm


class Conv2d(nn.Module):
    """This function implements 2d convolution.

    Arguments
    ---------
    out_channels : int
        It is the number of output channels.
    kernel_size : tuple
        Kernel size of the 2d convolutional filters over time and frequency
        axis.
    input_shape : tuple
        The shape of the input. Alternatively use ``in_channels``.
    in_channels : int
        The number of input channels. Alternatively use ``input_shape``.
    stride: int
        Stride factor of the 2d convolutional filters over time and frequency
        axis.
    dilation : int
        Dilation factor of the 2d convolutional filters over time and
        frequency axis.
    padding : str
        (same, valid). If "valid", no padding is performed.
        If "same" and stride is 1, output shape is same as input shape.
    padding_mode : str
        This flag specifies the type of padding. See torch.nn documentation
        for more information.
    groups : int
        This option specifies the convolutional groups. See torch.nn
        documentation for more information.
    bias : bool
        If True, the additive bias b is adopted.
    skip_transpose : bool
        If False, uses batch x time x channel convention of speechbrain.
        If True, uses batch x channel x time convention.
    weight_norm : bool
        If True, use weight normalization,
        to be removed with self.remove_weight_norm() at inference

    Example
    -------
    >>> inp_tensor = torch.rand([10, 40, 16, 8])
    >>> cnn_2d = Conv2d(
    ...     input_shape=inp_tensor.shape, out_channels=5, kernel_size=(7, 3)
    ... )
    >>> out_tensor = cnn_2d(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 40, 16, 5])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        input_shape=None,
        in_channels=None,
        stride=(1, 1),
        dilation=(1, 1),
        padding="same",
        groups=1,
        bias=True,
        padding_mode="reflect",
        skip_transpose=False,
        weight_norm=False,
    ):
        super().__init__()

        # handle the case if some parameter is int
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode
        self.unsqueeze = False
        self.skip_transpose = skip_transpose

        if input_shape is None and in_channels is None:
            raise ValueError("Must provide one of input_shape or in_channels")

        if in_channels is None:
            in_channels = self._check_input(input_shape)

        self.in_channels = in_channels

        # Weights are initialized following pytorch approach
        self.conv = nn.Conv2d(
            self.in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=groups,
            bias=bias,
        )

        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve. 2d or 4d tensors are expected.

        """
        if not self.skip_transpose:
            x = x.transpose(1, -1)

        if self.unsqueeze:
            x = x.unsqueeze(1)

        if self.padding == "same":
            x = self._manage_padding(
                x, self.kernel_size, self.dilation, self.stride
            )

        elif self.padding == "valid":
            pass

        else:
            raise ValueError(
                "Padding must be 'same' or 'valid'. Got " + self.padding
            )

        wx = self.conv(x)

        if self.unsqueeze:
            wx = wx.squeeze(1)

        if not self.skip_transpose:
            wx = wx.transpose(1, -1)

        return wx

    def _manage_padding(
        self,
        x,
        kernel_size: Tuple[int, int],
        dilation: Tuple[int, int],
        stride: Tuple[int, int],
    ):
        """This function performs zero-padding on the time and frequency axes
        such that their lengths is unchanged after the convolution.

        Arguments
        ---------
        x : torch.Tensor
        kernel_size : int
        dilation : int
        stride: int
        """
        # Detecting input shape
        L_in = self.in_channels

        # Time padding
        padding_time = get_padding_elem(
            L_in, stride[-1], kernel_size[-1], dilation[-1]
        )

        padding_freq = get_padding_elem(
            L_in, stride[-2], kernel_size[-2], dilation[-2]
        )
        padding = padding_time + padding_freq

        # Applying padding
        x = nn.functional.pad(x, padding, mode=self.padding_mode)

        return x

    def _check_input(self, shape):
        """Checks the input shape and returns the number of input channels.
        """

        if len(shape) == 3:
            self.unsqueeze = True
            in_channels = 1

        elif len(shape) == 4:
            in_channels = shape[3]

        else:
            raise ValueError("Expected 3d or 4d inputs. Got " + len(shape))

        # Kernel size must be odd
        if self.kernel_size[0] % 2 == 0 or self.kernel_size[1] % 2 == 0:
            raise ValueError(
                "The field kernel size must be an odd number. Got %s."
                % (self.kernel_size)
            )

        return in_channels

    def remove_weight_norm(self):
        """Removes weight normalization at inference if used during training.
        """
        self.conv = nn.utils.remove_weight_norm(self.conv)


class ConvolutionFrontEnd(Sequential):
    """This is a module to ensemble a convolution (depthwise) encoder with or
    without residual connection.

     Arguments
    ----------
    out_channels: int
        Number of output channels of this model (default 640).
    out_channels: Optional(list[int])
        Number of output channels for each of block.
    kernel_size: int
        Kernel size of convolution layers (default 3).
    strides: Optional(list[int])
        Striding factor for each block, this stride is applied at the last convolution layer at each block.
    num_blocks: int
        Number of block (default 21).
    num_per_layers: int
        Number of convolution layers for each block (default 5).
    dropout: float
        Dropout (default 0.15).
    activation: torch class
        Activation function for each block (default Swish).
    norm: torch class
        Normalization to regularize the model (default BatchNorm1d).
    residuals: Optional(list[bool])
        Whether apply residual connection at each block (default None).

    Example
    -------
    >>> x = torch.rand((8, 30, 10))
    >>> conv = ConvolutionFrontEnd(input_shape=x.shape)
    >>> out = conv(x)
    >>> out.shape
    torch.Size([8, 8, 3, 512])
    """

    def __init__(
        self,
        input_shape,
        num_blocks=3,
        num_layers_per_block=5,
        out_channels=[128, 256, 512],
        kernel_sizes=[3, 3, 3],
        strides=[1, 2, 2],
        dilations=[1, 1, 1],
        residuals=[True, True, True],
        conv_module=Conv2d,
        activation=torch.nn.LeakyReLU,
        norm=LayerNorm,
        dropout=0.1,
    ):
        super().__init__(input_shape=input_shape)
        for i in range(num_blocks):
            self.append(
                ConvBlock,
                num_layers=num_layers_per_block,
                out_channels=out_channels[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                dilation=dilations[i],
                residual=residuals[i],
                conv_module=conv_module,
                activation=activation,
                norm=norm,
                dropout=dropout,
                layer_name=f"convblock_{i}",
            )


class ConvBlock(torch.nn.Module):
    """An implementation of convolution block with 1d or 2d convolutions (depthwise).

    Arguments
    ----------
    out_channels : int
        Number of output channels of this model (default 640).
    kernel_size : int
        Kernel size of convolution layers (default 3).
    strides : int
        Striding factor for this block (default 1).
    num_layers : int
        Number of depthwise convolution layers for this block.
    activation : torch class
        Activation function for this block.
    norm : torch class
        Normalization to regularize the model (default BatchNorm1d).
    residuals: bool
        Whether apply residual connection at this block (default None).

    Example
    -------
    >>> x = torch.rand((8, 30, 10))
    >>> conv = ConvBlock(2, 16, input_shape=x.shape)
    >>> out = conv(x)
    >>> x.shape
    torch.Size([8, 30, 10])
    """

    def __init__(
        self,
        num_layers,
        out_channels,
        input_shape,
        kernel_size=3,
        stride=1,
        dilation=1,
        residual=False,
        conv_module=Conv2d,
        activation=torch.nn.LeakyReLU,
        norm=None,
        dropout=0.1,
    ):
        super().__init__()

        self.convs = Sequential(input_shape=input_shape)

        for i in range(num_layers):
            self.convs.append(
                conv_module,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride if i == num_layers - 1 else 1,
                dilation=dilation,
                layer_name=f"conv_{i}",
            )
            if norm is not None:
                self.convs.append(norm, layer_name=f"norm_{i}")
            self.convs.append(activation(), layer_name=f"act_{i}")
            self.convs.append(
                torch.nn.Dropout(dropout), layer_name=f"dropout_{i}"
            )

        self.reduce_conv = None
        self.drop = None
        if residual:
            self.reduce_conv = Sequential(input_shape=input_shape)
            self.reduce_conv.append(
                conv_module,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                layer_name="conv",
            )
            self.reduce_conv.append(norm, layer_name="norm")
            self.drop = torch.nn.Dropout(dropout)

    def forward(self, x):
        """ Processes the input tensor x and returns an output tensor."""
        out = self.convs(x)
        if self.reduce_conv:
            out = out + self.reduce_conv(x)
            out = self.drop(out)

        return out


def get_padding_elem(L_in: int, stride: int, kernel_size: int, dilation: int):
    """This function computes the number of elements to add for zero-padding.

    Arguments
    ---------
    L_in : int
    stride: int
    kernel_size : int
    dilation : int
    """
    if stride > 1:
        padding = [math.floor(kernel_size / 2), math.floor(kernel_size / 2)]

    else:
        L_out = (
            math.floor((L_in - dilation * (kernel_size - 1) - 1) / stride) + 1
        )
        padding = [
            math.floor((L_in - L_out) / 2),
            math.floor((L_in - L_out) / 2),
        ]
    return padding


def get_padding_elem_transposed(
    L_out: int,
    L_in: int,
    stride: int,
    kernel_size: int,
    dilation: int,
    output_padding: int,
):
    """This function computes the required padding size for transposed convolution

    Arguments
    ---------
    L_out : int
    L_in : int
    stride: int
    kernel_size : int
    dilation : int
    output_padding : int
    """

    padding = -0.5 * (
        L_out
        - (L_in - 1) * stride
        - dilation * (kernel_size - 1)
        - output_padding
        - 1
    )
    return int(padding)
