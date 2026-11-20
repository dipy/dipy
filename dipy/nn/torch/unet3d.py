"""PyTorch version of the 3D U-Net of the DeepN4 and Synb0 models.

``dipy.nn.utils.load_unet3d`` builds it to run these models on a GPU. The\nCPU path never imports this module, nor PyTorch.
"""

from dipy.utils.optpkg import optional_package

torch, have_torch, _ = optional_package("torch", min_version="2.2.0")
if have_torch:
    from torch.nn import (
        Conv3d,
        ConvTranspose3d,
        InstanceNorm3d,
        LeakyReLU,
        MaxPool3d,
        Module,
        Sequential,
    )
else:

    class Module:
        pass


class UNet3D(Module):
    """3D U-Net architecture of the DeepN4 and Synb0 models.

    Parameters
    ----------
    n_in : int
        Number of input channels.
    n_out : int
        Number of output channels.
    """

    def __init__(self, n_in, n_out):
        super().__init__()
        # Encoder
        c = 32
        self.ec0 = self.encoder_block(n_in, c, kernel_size=3, stride=1, padding=1)
        self.ec1 = self.encoder_block(c, c * 2, kernel_size=3, stride=1, padding=1)
        self.pool0 = MaxPool3d(2)
        self.ec2 = self.encoder_block(c * 2, c * 2, kernel_size=3, stride=1, padding=1)
        self.ec3 = self.encoder_block(c * 2, c * 4, kernel_size=3, stride=1, padding=1)
        self.pool1 = MaxPool3d(2)
        self.ec4 = self.encoder_block(c * 4, c * 4, kernel_size=3, stride=1, padding=1)
        self.ec5 = self.encoder_block(c * 4, c * 8, kernel_size=3, stride=1, padding=1)
        self.pool2 = MaxPool3d(2)
        self.ec6 = self.encoder_block(c * 8, c * 8, kernel_size=3, stride=1, padding=1)
        self.ec7 = self.encoder_block(c * 8, c * 16, kernel_size=3, stride=1, padding=1)
        self.el = Conv3d(c * 16, c * 16, kernel_size=1, stride=1, padding=0)

        # Decoder
        self.dc9 = self.decoder_block(
            c * 16, c * 16, kernel_size=2, stride=2, padding=0
        )
        self.dc8 = self.decoder_block(
            c * 16 + c * 8, c * 8, kernel_size=3, stride=1, padding=1
        )
        self.dc7 = self.decoder_block(c * 8, c * 8, kernel_size=3, stride=1, padding=1)
        self.dc6 = self.decoder_block(c * 8, c * 8, kernel_size=2, stride=2, padding=0)
        self.dc5 = self.decoder_block(
            c * 8 + c * 4, c * 4, kernel_size=3, stride=1, padding=1
        )
        self.dc4 = self.decoder_block(c * 4, c * 4, kernel_size=3, stride=1, padding=1)
        self.dc3 = self.decoder_block(c * 4, c * 4, kernel_size=2, stride=2, padding=0)
        self.dc2 = self.decoder_block(
            c * 4 + c * 2, c * 2, kernel_size=3, stride=1, padding=1
        )
        self.dc1 = self.decoder_block(c * 2, c * 2, kernel_size=3, stride=1, padding=1)
        self.dc0 = self.decoder_block(c * 2, n_out, kernel_size=1, stride=1, padding=0)
        self.dl = ConvTranspose3d(n_out, n_out, kernel_size=1, stride=1, padding=0)

    def encoder_block(self, in_channels, out_channels, kernel_size, stride, padding):
        """Encoder block for the 3D U-Net.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int
            Size of the convolutional kernel.
        stride : int
            Stride of the convolution.
        padding : int
            Padding of the convolution.

        Returns
        -------
        torch.nn.Sequential
            The encoder block.
        """
        layer = Sequential(
            Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            InstanceNorm3d(out_channels),
            LeakyReLU(),
        )
        return layer

    def decoder_block(self, in_channels, out_channels, kernel_size, stride, padding):
        """Decoder block for the 3D U-Net.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int
            Size of the convolutional kernel.
        stride : int
            Stride of the convolution.
        padding : int
            Padding of the convolution.

        Returns
        -------
        torch.nn.Sequential
            The decoder block.
        """
        layer = Sequential(
            ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            InstanceNorm3d(out_channels),
            LeakyReLU(),
        )
        return layer

    def forward(self, x):
        """Forward pass of the 3D U-Net.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        # Encodes
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        del e0

        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        syn2 = self.ec5(e4)
        del e3, e4

        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        e7 = self.ec7(e6)

        # Last layer without relu
        el = self.el(e7)
        del e5, e6, e7

        # Decode
        d9 = torch.cat((self.dc9(el), syn2), 1)
        del el, syn2

        d8 = self.dc8(d9)
        d7 = self.dc7(d8)
        del d9, d8

        d6 = torch.cat((self.dc6(d7), syn1), 1)
        del d7, syn1

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        del d6, d5

        d3 = torch.cat((self.dc3(d4), syn0), 1)
        del d4, syn0

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2

        d0 = self.dc0(d1)
        del d1

        # Last layer without relu
        out = self.dl(d0)
        return out
