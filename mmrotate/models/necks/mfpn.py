import math
import mmengine
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmengine.model import ModuleList
from mmengine.model.weight_init import trunc_normal_
from mmpretrain.models import build_2d_sincos_position_embedding

from transformers.models.mamba.modeling_mamba import MambaMixer

from mmrotate.registry import MODELS
from mmpretrain.models.utils import build_norm_layer, to_2tuple
from mmdet.models.necks import FPN
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule
from typing import List, Tuple, Union
from torch import Tensor
from mmrotate.models.component import DWConv, DWSConv


@MODELS.register_module()
class MFPN(FPN):
    def __init__(
        self,
        img_size,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        add_extra_convs=False,
        relu_before_extra_convs=False,
        no_norm_on_lateral=False,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform')
    ):
        super(MFPN, self).__init__(
            in_channels,
            out_channels,
            num_outs,
            start_level,
            end_level,
            add_extra_convs,
            relu_before_extra_convs,
            no_norm_on_lateral,
            conv_cfg,
            norm_cfg,
            act_cfg,
            init_cfg=init_cfg
        )

        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            fpn_conv = DWSConv(
                out_channels,
                out_channels,
                3
            )
            self.fpn_convs.append(fpn_conv)

        self.mlayers = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level-1):
            # patch_cfg = dict(stride=1)
            init_cfg = [
                dict(
                    type='Kaiming',
                    layer='Conv2d',
                    mode='fan_in',
                    nonlinearity='linear'
                )
            ]

            mlayer = MLayer(
                num_layers=1,
                in_channels=256,
                img_size=img_size//(8 * 2**i),
                embed_dims=256,
                # patch_size=8//(2**i) if (8//(2**i))>=1 else 1,
                patch_size=1,
                pe_type='learnable',
                # pe_type='sine',
                drop_rate=0.,
                norm_cfg=dict(type='LN', eps=1e-6),
                patch_cfg=dict(),
                layer_cfgs=dict(),
                init_cfg=init_cfg)
            self.mlayers.append(mlayer)


    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        used_backbone_levels = len(laterals)

        # mamba layers
        for i in range(0, used_backbone_levels-1):
            laterals[i+1] = self.mlayers[i](laterals[i+1])
        # build top-down path

        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


class MLayer(BaseModule):
    def __init__(self,
                 num_layers=2,
                 in_channels=256,
                 img_size=1024,
                 embed_dims=256,
                 patch_size=16,
                 pe_type='learnable',
                 drop_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 init_cfg=None):
        super(MLayer, self).__init__(init_cfg)

        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.img_size = to_2tuple(img_size)

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set position embedding
        self.pe_type = pe_type
        self.interpolate_mode = 'bicubic'
        if pe_type == 'learnable':
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dims))
        elif pe_type == 'sine':
            self.pos_embed = build_2d_sincos_position_embedding(
                patches_resolution=self.patch_resolution,
                embed_dims=self.embed_dims,
                temperature=10000,
                cls_token=False)
        else:
            self.pos_embed = None

        # Set drop after position embedding
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        self.layers = ModuleList()
        for i in range(self.num_layers):
            self.layers.append(
                MBlock(
                    i,
                    embed_dims=embed_dims,
                    layer_cfgs=layer_cfgs,
                    norm_cfg=norm_cfg
                )
            )
        self.init_weights()

    def init_weights(self):
        super(MLayer, self).init_weights()
        if not (isinstance(self.init_cfg, dict) and self.init_cfg['type'] == 'Pretrained'):
            if self.pos_embed is not None:
                trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, C, _, _ = x.shape

        # patch embed
        x, patch_resolution = self.patch_embed(x)
        # pos embed
        if self.pos_embed is not None:
            x = x + self.pos_embed.to(device=x.device)
        # drop
        x = self.drop_after_pos(x)
        # mamba layer
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = x.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])

        return x



class MBlock(BaseModule):
    def __init__(
            self,
            layer_idx,
            embed_dims=256,
            layer_cfgs=dict(),
            norm_cfg=dict(type='LN', eps=1e-6)
    ):
        super(MBlock, self).__init__()
        self.embed_dims = embed_dims
        _layer_cfg = dict(
            hidden_size=self.embed_dims,
            state_size=16,
            intermediate_size=self.embed_dims * 2,
            conv_kernel=4,
            time_step_rank=math.ceil(self.embed_dims / 16),
            use_conv_bias=True,
            hidden_act="silu",
            use_bias=False,
        )
        _layer_cfg.update(layer_cfgs)
        _layer_cfg = mmengine.Config(_layer_cfg)
        self.mamba = MambaMixer(_layer_cfg, layer_idx)
        self.gate = nn.Sequential(
                nn.Linear(3 * self.embed_dims, 3, bias=False),
                nn.Softmax(dim=-1)
        )
        self.pre_norm = build_norm_layer(norm_cfg, self.embed_dims)

    def forward(self, x):
        B = x.shape[0]
        residual = x

        x_inputs = [x, torch.flip(x, [1])]
        rand_index = torch.randperm(x.size(1))
        x_inputs.append(x[:, rand_index])
        x_inputs = torch.cat(x_inputs, dim=0)
        x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))
        x_inputs = self.mamba(x_inputs)
        forward_x, reverse_x, shuffle_x = torch.split(x_inputs, B, dim=0)
        reverse_x = torch.flip(reverse_x, [1])
        # reverse the random index
        rand_index = torch.argsort(rand_index)
        shuffle_x = shuffle_x[:, rand_index]
        mean_forward_x = torch.mean(forward_x, dim=1)
        mean_reverse_x = torch.mean(reverse_x, dim=1)
        mean_shuffle_x = torch.mean(shuffle_x, dim=1)
        gate = torch.cat([mean_forward_x, mean_reverse_x, mean_shuffle_x], dim=-1)
        gate = self.gate(gate)
        gate = gate.unsqueeze(-1)
        x = gate[:, 0:1] * forward_x + gate[:, 1:2] * reverse_x + gate[:, 2:3] * shuffle_x

        return residual + x