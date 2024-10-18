import re
import time
import math
import numpy as np
from functools import partial
from typing import Optional, Union, Type, List, Tuple, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from mlpipeline.models.segmentation.swin_umamba.network_initialization import InitWeights_He
from mlpipeline.models.segmentation.swin_umamba.original_modules import SS2D
from mlpipeline.models.segmentation.swin_umamba.k1_modules_2d import SS2DepthConv_K1, SS2Depth_K1, SS2DConv_K1


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    Reference: http://arxiv.org/abs/2401.10166
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchExpand(nn.Module):
    """
    Reference: https://arxiv.org/pdf/2105.05537.pdf
    """
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        # B, C, H, W ==> B, H, W, C
        x = x.permute(0, 2, 3, 1)
        x = self.expand(x)
        B, H, W, C = x.shape

        x = x.view(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c) -> b (h p1) (w p2) c", p1=2, p2=2, c=C//4)
        x = x.view(B, -1, C//4)
        x = self.norm(x)
        x = x.reshape(B, H*2, W*2, C//4)

        return x


class FinalPatchExpand_X4(nn.Module):
    """
    Reference:
        - GitHub: https://github.com/HuCaoFighting/Swin-Unet/blob/main/networks/swin_transformer_unet_skip_expand_decoder_sys.py
        - Paper: https://arxiv.org/pdf/2105.05537.pdf
    """
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        # self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # H, W = self.input_resolution
        # B, C, H, W ==> B, H, W, C
        x = x.permute(0, 2, 3, 1)
        x = self.expand(x)
        B, H, W, C = x.shape
        # B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        # x = x.view(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        x = x.reshape(B, H*self.dim_scale, W*self.dim_scale, self.output_dim)

        # x.permute(0, 3, 1, 2)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        use_conv: bool = False,
        use_depth: bool = False,
        mode: str = "",
        d_depth_stride: int = 16,
        d_depth_out: int = 1,
        d_depth_squeeze: int = 1,
        conv_mode: str = "",
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        if use_conv and use_depth:
            self.self_attention = SS2DepthConv_K1(
                d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state,
                mode=mode,
                d_depth_stride=d_depth_stride,
                d_depth_out=d_depth_out,
                d_depth_squeeze=d_depth_squeeze,
                conv_mode=conv_mode,
                **kwargs)
        elif use_depth:
            self.self_attention = SS2Depth_K1(
                d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state,
                mode=mode,
                d_depth_stride=d_depth_stride,
                d_depth_out=d_depth_out,
                d_depth_squeeze=d_depth_squeeze,
                **kwargs)
        elif use_conv:
            self.self_attention = SS2DConv_K1(
                d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state,
                conv_mode=conv_mode,
                **kwargs)
        else:
            self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        depth,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        d_state=16,
        use_conv=False,
        use_depth=False,
        mode="",
        d_depth_stride=16,
        d_depth_out=1,
        d_depth_squeeze=1,
        conv_mode="",
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                use_conv=use_conv,
                use_depth=use_depth,
                mode=mode,
                d_depth_stride=d_depth_stride,
                d_depth_out=d_depth_out,
                d_depth_squeeze=d_depth_squeeze,
                conv_mode=conv_mode,
            )
            for i in range(depth)])

        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class VSSMEncoder(nn.Module):
    def __init__(
            self, patch_size=4, in_chans=3, depths=[2, 2, 9, 2],
            dims=[96, 192, 384, 768], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
            norm_layer=nn.LayerNorm, patch_norm=True,
            use_checkpoint=False, **kwargs,
        ):
        super().__init__()
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        # WASTED absolute position embedding ======================
        self.ape = False
        # self.ape = False
        # drop_rate = 0.0
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                # downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                downsample=None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                self.downsamples.append(PatchMerging2D(dim=dims[i_layer], norm_layer=norm_layer))

        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless

        Conv2D is not intialized !!!
        """
        # print(m, getattr(getattr(m, "weight", nn.Identity()), "INIT", None), isinstance(m, nn.Linear), "======================")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward(self, x):
        x_ret = []
        x_ret.append(x)

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for s, layer in enumerate(self.layers):
            x = layer(x)
            x_ret.append(x.permute(0, 3, 1, 2))
            if s < len(self.downsamples):
                x = self.downsamples[s](x)

        return x_ret


class UNetResDecoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        deep_supervision,
        mode: str,
        depth_stride_configs: List[int],
        depth_stride_output_sizes: List[int],
        depth_squeeze_factors: List[int],
        conv_mode: str,
        features_per_stage: Union[Tuple[int, ...], List[int]] = None,
        drop_path_rate: float = 0.2,
        d_state: int = 16,
        use_convs: List[bool] = None,
        use_depths: List[bool] = None,
    ):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()

        encoder_output_channels = features_per_stage
        self.deep_supervision = deep_supervision
        # self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder_output_channels)

        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, (n_stages_encoder-1)*2)]
        depths = [2, 2, 2, 2]

        # we start with the bottleneck and work out way up
        stages = []
        expand_layers = []
        seg_layers = []
        concat_back_dim = []

        for s in range(1, n_stages_encoder):
            input_features_below = encoder_output_channels[-s]
            input_features_skip = encoder_output_channels[-(s + 1)]
            expand_layers.append(PatchExpand(
                input_resolution=None,
                dim=input_features_below,
                dim_scale=2,
                norm_layer=nn.LayerNorm,
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(VSSLayer(
                dim=input_features_skip,
                depth=2,
                attn_drop=0.0,
                drop_path=dpr[sum(depths[:s-1]):sum(depths[:s])],
                d_state=math.ceil(2*input_features_skip / 6) if d_state is None else d_state,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=False,
                use_conv=use_convs[s - 1],
                use_depth=use_depths[s - 1],
                mode=mode,
                d_depth_stride=depth_stride_configs[s - 1],
                d_depth_out=depth_stride_output_sizes[s - 1],
                d_depth_squeeze=depth_squeeze_factors[s - 1],
                conv_mode=conv_mode,
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(nn.Conv2d(input_features_skip, num_classes, 1, 1, 0, bias=True))
            concat_back_dim.append(nn.Linear(2*input_features_skip, input_features_skip))

        # for final prediction
        expand_layers.append(FinalPatchExpand_X4(
            input_resolution=None,
            dim=encoder_output_channels[0],
            dim_scale=4,
            norm_layer=nn.LayerNorm,
        ))
        stages.append(nn.Identity())
        seg_layers.append(nn.Conv2d(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.expand_layers = nn.ModuleList(expand_layers)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.concat_back_dim = nn.ModuleList(concat_back_dim)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.expand_layers[s](lres_input)
            if s < (len(self.stages) - 1):
                x = torch.cat((x, skips[-(s+2)].permute(0, 2, 3, 1)), -1)
                x = self.concat_back_dim[s](x)

            x = self.stages[s](x).permute(0, 3, 1, 2)
            # print(s, x.shape, len(self.stages[s].blocks) if hasattr(self.stages[s], "blocks") else 0)

            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r


class SwinUMambaD(nn.Module):
    def __init__(self, vss_args, decoder_args):
        super().__init__()
        self.vssm_encoder = VSSMEncoder(**vss_args)
        self.decoder = UNetResDecoder(**decoder_args)

    def forward(self, x):
        skips = self.vssm_encoder(x)
        out = self.decoder(skips)
        return out

    @torch.no_grad()
    def freeze_encoder(self):
        for name, param in self.vssm_encoder.named_parameters():
            if "patch_embed" not in name:
                param.requires_grad = False

    @torch.no_grad()
    def unfreeze_encoder(self):
        for param in self.vssm_encoder.parameters():
            param.requires_grad = True


def load_pretrained_ckpt(
    model,
    num_input_channels=1,
    ckpt_path = "./data/pretrained/vmamba/vmamba_tiny_e292.pth"
):

    print(f"Loading weights from: {ckpt_path}")
    skip_params = ["norm.weight", "norm.bias", "head.weight", "head.bias"]

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_dict = model.state_dict()
    for k, v in ckpt["model"].items():
        if k in skip_params:
            print(f"Skipping weights: {k}")
            continue
        kr = f"vssm_encoder.{k}"
        if "patch_embed" in k and ckpt["model"]["patch_embed.proj.weight"].shape[1] != num_input_channels:
            print(f"Passing weights: {k}")
            continue
        if "downsample" in kr:
            i_ds = int(re.findall(r"layers\.(\d+)\.downsample", kr)[0])
            kr = kr.replace(f"layers.{i_ds}.downsample", f"downsamples.{i_ds}")
            assert kr in model_dict.keys()
        if kr in model_dict.keys():
            assert v.shape == model_dict[kr].shape, f"Shape mismatch: {v.shape} vs {model_dict[kr].shape}"
            model_dict[kr] = v
        else:
            print(f"Passing weights: {k}")

    model.load_state_dict(model_dict)

    return model


def get_swin_umamba_from_plans(
    dataset_json: dict,
    num_input_channels: int,
    deep_supervision: bool = True,
    use_pretrain: bool = False,
):
    dim = 2
    assert dim == 2, "Only 2D supported at the moment"

    n_stages_encoder = 4
    no_squeeze = False
    patch_size = (512, 512)
    squeeze_factor = 2

    if patch_size[0] == 384 and patch_size[1] == 640:
        depth_stride_configs = [2] * (n_stages_encoder - 2) + [4]
        depth_stride_output_sizes = [240, 960, 960]
        depth_squeeze_factors = [1, 1, 1] if no_squeeze else [squeeze_factor, 1, 1]
    elif patch_size[0] == 512 and patch_size[1] == 512:
        depth_stride_configs = [2] * (n_stages_encoder - 2) + [4]
        depth_stride_output_sizes = [256, 1024, 1024]
        depth_squeeze_factors = [1, 1, 1] if no_squeeze else [squeeze_factor, 1, 1]

    vss_args = dict(
        in_chans=num_input_channels,
        patch_size=4,
        depths=[2, 2, 9, 2],
        dims=96,
        drop_path_rate=0.2,
    )

    decoder_args = dict(
        num_classes=1,
        deep_supervision=deep_supervision,
        features_per_stage=[96, 192, 384, 768],
        drop_path_rate=0.2,
        d_state=16,
        use_convs=[False, False, False],
        use_depths=[False, False, False],
        mode="even",
        depth_stride_configs=depth_stride_configs,
        depth_stride_output_sizes=depth_stride_output_sizes,
        depth_squeeze_factors=depth_squeeze_factors,
        conv_mode="full",
    )

    model = SwinUMambaD(vss_args, decoder_args)
    model.apply(InitWeights_He(1e-2))
    model.apply(init_last_bn_before_add_to_0)

    if use_pretrain:
        model = load_pretrained_ckpt(model, num_input_channels=num_input_channels)

    return model
