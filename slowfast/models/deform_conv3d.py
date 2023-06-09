import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from detectron2.layers.deform_conv import modulated_deform_conv
from detectron2.layers.wrappers import Conv2d, _NewEmptyTensorOp


class ModulatedDeformConv3DWithOffset(nn.Module):
    """
    ModulatedDeformConv3D, only support Temporal Kernel = 1.
    It's a workaround to support Tensor (B, C, T, H, W).

    Besides, encapsulate Conv_offset here
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=True,
        norm=None,
        activation=None,
    ):
        """
        Modulated deformable convolution from :paper:`deformconv2`.
        Arguments are similar to :class:`Conv2D`. Extra arguments:
        Args:
            deformable_groups (int): number of groups used in deformable convolution.
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        """
        super(ModulatedDeformConv3DWithOffset, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.norm = norm
        self.activation = activation

        """
        log msg: !! res5_0_branch2b_w: (512, 512, 1, 3, 3) does not match s5.pathway0_res0.branch2.b.weight: (512, 512, 3, 3)
        define weight with temporal dimension to make loding pretrain model happy.
        """
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, 1, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

        assert not (norm or activation), "add Activation adn BatchNorm outside for loading pretrain model"
        # Add conv_offset below:
        self.conv_offset = Conv2d(
            in_channels,
            27 * 1,  # only support Modulated DeformConv and num_groups = 1 now
            kernel_size=3,
            stride=stride,
            padding=1 * dilation,
            dilation=dilation,
        )
        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)

    def forward(self, x):
        B, _, T = x.shape[:3]
        x = x.permute(0, 2, 1, 3, 4).flatten(start_dim=0, end_dim=1).contiguous()  # -> (B*T, C, H, W)

        offset_mask = self.conv_offset(x)
        offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((offset_x, offset_y), dim=1)
        mask = mask.sigmoid()

        # below is the original `ModulatedDeformConv` forward, without any modification
        if x.numel() == 0:
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            return _NewEmptyTensorOp.apply(x, output_shape)

        x = modulated_deform_conv(
            x,
            offset,
            mask,
            self.weight.squeeze(dim=2),  # squeeze temporal dimension
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deformable_groups,
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)

        # now, reverse the shape to (B, C', T, H', W')
        _, C, H, W = x.shape
        x = x.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        return x

    def extra_repr(self):
        tmpstr = "in_channels=" + str(self.in_channels)
        tmpstr += ", out_channels=" + str(self.out_channels)
        tmpstr += ", kernel_size=" + str(self.kernel_size)
        tmpstr += ", stride=" + str(self.stride)
        tmpstr += ", padding=" + str(self.padding)
        tmpstr += ", dilation=" + str(self.dilation)
        tmpstr += ", groups=" + str(self.groups)
        tmpstr += ", deformable_groups=" + str(self.deformable_groups)
        tmpstr += ", bias=" + str(self.with_bias)
        return tmpstr
