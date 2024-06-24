import collections.abc
from itertools import repeat
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import DropPath
from timm.models.registry import register_model

from mlpipeline.models.segmentation.eoformer.EoM import EoS, EoL
from mlpipeline.models.segmentation.eoformer.unet import UnetrUpBlock_light, UpBlock_light


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)


class Downsampling(nn.Module):
    def __init__(self, in_channels, out_channels,
        kernel_size, stride=1, padding=0,
        pre_norm=None, post_norm=None, pre_permute=False):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding)
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x): # pre norm --> (permute) --> downsample(conv) --> post norm
        x = self.pre_norm(x) # input shape: [B, C, H, W, D]
        if self.pre_permute:
            # if take [B, H, W, C] as input, permute it to [B, C, H, W]
            x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.conv(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous() # # [B, C, H, W, D] -> # [B, H, W, D, C]
        x = self.post_norm(x)
        return x

class LayerNormGeneral(nn.Module):
    def __init__(self, affine_shape=None, normalized_dim=(-1, ), scale=True, # normalized_dim = -1 代表对最后一个维度即 channel, 每一个token做归一化，使得数据均值为0方差为1
        bias=True, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x

class Scale(nn.Module):
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale

class SquaredReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
    def forward(self, x):
        return torch.square(self.relu(x))

class StarReLU(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True,
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias

class Pooling(nn.Module):
    def __init__(self, pool_size=3, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool3d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        y = x.permute(0, 4, 1, 2, 3).contiguous() # input shape: [B, H, W, D, C]
        y = self.pool(y)
        y = y.permute(0, 2, 3, 4, 1).contiguous()
        return y - x


class Attention(nn.Module):
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
        attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        B, H, W, D, C = x.shape
        N = H * W * D
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, D, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EfficientAttention(nn.Module):
    def __init__(self, dim, head_count=1, **kwargs):
        super().__init__()
        self.in_channels = dim
        self.key_channels = dim
        self.head_count = head_count
        self.value_channels = dim

        self.keys = nn.Conv3d(self.in_channels, self.key_channels, 1)
        self.queries = nn.Conv3d(self.in_channels, self.key_channels, 1)
        self.values = nn.Conv3d(self.in_channels, self.value_channels, 1)
        self.reprojection = nn.Conv3d(self.value_channels, self.in_channels, 1)

    def forward(self, input_): #input : [B, H, W, D, C]
        input_ = input_.permute(0, 4, 1, 2, 3) # [B, H, W, D, C] --> [B, C, H, W, D]
        n, _, h, w, d  = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w * d))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w * d)
        values = self.values(input_).reshape((n, self.value_channels, h * w * d))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2)

            query = F.softmax(queries[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=1)

            value = values[:, i * head_value_channels : (i + 1) * head_value_channels, :]

            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w, d)  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)
        attention = attention.permute(0, 2, 3, 4, 1) # [B, C, H, W, D] --> [B, H, W, D, C]
        return attention


class SepConv(nn.Module):
    def __init__(self, dim, expansion_ratio=2,
        act1_layer=StarReLU, act2_layer=nn.Identity,
        bias=False, kernel_size=3, padding=1,
        **kwargs, ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv3d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias) # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x): # fc(dim --> dim*expansion_ratio) --> act1 --> DWConv --> act2 --> fc(dim*expansion_ratio --> dim)
        x = self.pwconv1(x) # input shape: [B, H, W, D, C]
        x = self.act1(x)
        x = x.permute(0, 4, 1, 2, 3) # [B, H, W, D, C] --> [B, C, H, W, D]
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1) # [B, C, H, W, D] --> [B, H, W, D, C]
        x = self.act2(x)
        x = self.pwconv2(x)
        return x

class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0., bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

DOWNSAMPLE_LAYERS_FOUR_STAGES = [partial(Downsampling,
            kernel_size=3, stride=2, padding=1,
            post_norm=partial(LayerNormGeneral, bias=False, eps=1e-6)
            )] + \
            [partial(Downsampling,
                kernel_size=3, stride=2, padding=1,
                pre_norm=partial(LayerNormGeneral, bias=False, eps=1e-6), pre_permute=True
            )]*3

class Block(nn.Module):
    def __init__(self, dim,
                token_mixer=nn.Identity,
                mlp=Mlp,
                norm_layer=nn.LayerNorm,
                drop=0., drop_path=0.,
                layer_scale_init_value=None, res_scale_init_value=None,
                scale_trainable=True,
                position_embedding=None,
                ):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, drop=drop, act_type='prelu', with_idt=True, head_count=dim//32)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value, trainable=scale_trainable) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value, trainable=scale_trainable) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()
        self.position_embedding = position_embedding

    def forward(self, x):
        if self.position_embedding:
            x = self.res_scale1(x) + \
                self.layer_scale1(
                    self.drop_path1(
                        self.token_mixer(self.norm1(self.position_embedding(x)))
                    )
                )
        else:
            x = self.res_scale1(x) + \
                self.layer_scale1(
                    self.drop_path1(
                        self.token_mixer(self.norm1(x))
                    )
                )
        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(x))
                )
            )
        return x


class Encoder(nn.Module):
    def __init__(self, in_chans=4,
                depths=[2, 2, 2, 2],
                dims=[32, 64, 128, 256],
                downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
                token_mixers=nn.Identity,
                mlps=Mlp,
                norm_layers=partial(LayerNormGeneral, eps=1e-6, bias=False),
                drop_path_rate=0.1,
                head_dropout=0.0,
                layer_scale_init_values=None,
                res_scale_init_values=[None, None, 1.0, 1.0],
                position_embeddings=[[None, None], [None, None], [None, None], [None, None]],
                output_norm=partial(nn.LayerNorm, eps=1e-6),
                **kwargs,
                ):
        super().__init__()

        if not isinstance(depths, (list, tuple)):
            depths = [depths]
        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        num_stage = len(depths)
        self.num_stage = num_stage

        if not isinstance(downsample_layers, (list, tuple)):
            downsample_layers = [downsample_layers] * num_stage
        down_dims = [in_chans] + dims # [4, 32, 64, 128, 256]
        self.downsample_layers = nn.ModuleList(
            [downsample_layers[i](down_dims[i], down_dims[i+1]) for i in range(num_stage)]
        )

        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage

        if not isinstance(mlps, (list, tuple)):
            mlps = [mlps] * num_stage

        if not isinstance(norm_layers, (list, tuple)):
            norm_layers = [norm_layers] * num_stage

        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if not isinstance(layer_scale_init_values, (list, tuple)):
            layer_scale_init_values = [layer_scale_init_values] * num_stage
        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * num_stage

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(num_stage):
            stage = nn.Sequential(
                *[Block(dim=dims[i],
                token_mixer=token_mixers[i],
                mlp=mlps[i],
                norm_layer=norm_layers[i],
                drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_values[i],
                res_scale_init_value=res_scale_init_values[i],
                position_embedding=position_embeddings[i][j],
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'}

    def forward_features(self, x):
        intmd_output = {}

        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            intmd_output[i] = x.permute(0, 4, 1, 2, 3).contiguous() # (B, H, W, D, C) -> (B, C, H, W, D)
        return intmd_output[3], intmd_output

    def forward(self, x):
        x, intmd_output = self.forward_features(x)
        return x, intmd_output


UPSAMPLE_LAYERS_FOUR_STAGES = [partial(UnetrUpBlock_light,
            spatial_dims=3,
            kernel_size=3, upsample_kernel_size=2,
            norm_name=("group", {"num_groups": 1}),
            )] * 3 + \
            [partial(UpBlock_light,
            spatial_dims=3,
            kernel_size=3, upsample_kernel_size=2,
            norm_name=("group", {"num_groups": 1}),
            )]


class Decoder(nn.Module):
    def __init__(self, in_chans=256,
                depths=[2, 2, 2],
                dims=[128, 64, 32, 16],
                upsample_layers=UPSAMPLE_LAYERS_FOUR_STAGES,
                token_mixers=nn.Identity,
                mlps=Mlp,
                norm_layers=partial(LayerNormGeneral, eps=1e-6, bias=False),
                drop_path_rate=0.1,
                head_dropout=0.0,
                layer_scale_init_values=None,
                res_scale_init_values=[None, None, 1.0, 1.0],
                position_embeddings=[[None, None], [None, None], [None, None], [None, None]],
                output_norm=partial(nn.LayerNorm, eps=1e-6),
                **kwargs,
                ):
        super().__init__()

        if not isinstance(depths, (list, tuple)):
            depths = [depths] # it means the model has only one stage
        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        num_stage = len(depths)
        self.num_stage = num_stage

        if not isinstance(upsample_layers, (list, tuple)):
            upsample_layers = [upsample_layers] * num_stage
        up_dims = [in_chans] + dims # [256, 128, 64, 32, 16]
        self.upsample_layers = nn.ModuleList(
            [upsample_layers[i](in_channels=up_dims[i], out_channels=up_dims[i+1]) for i in range(num_stage+1)]
        )

        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage

        if not isinstance(mlps, (list, tuple)):
            mlps = [mlps] * num_stage

        if not isinstance(norm_layers, (list, tuple)):
            norm_layers = [norm_layers] * num_stage

        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))][::-1]

        if not isinstance(layer_scale_init_values, (list, tuple)):
            layer_scale_init_values = [layer_scale_init_values] * num_stage
        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * num_stage

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(num_stage):
            stage = nn.Sequential(
                *[Block(dim=dims[i],
                token_mixer=token_mixers[i][j],
                mlp=mlps[i],
                norm_layer=norm_layers[i],
                drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_values[i],
                res_scale_init_value=res_scale_init_values[i],
                position_embedding=position_embeddings[i][j],
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'}

    def forward_features(self, intmd_output):
        x = intmd_output[3]
        for i in range(self.num_stage):
            x = self.upsample_layers[i](x, intmd_output[2-i]) # (B, C, H, W, D)
            x = x.permute(0, 2, 3, 4, 1) # (B, C, H, W, D) -> (B, H, W, D, C)
            x = self.stages[i](x)
            x = x.permute(0, 4, 1, 2, 3) # (B, H, W, D, C) -> (B, C, H, W, D)
        x = self.upsample_layers[-1](x) # (B, C, H, W, D)
        return x

    def forward(self, intmd_output):
        out = self.forward_features(intmd_output)
        return out


@register_model
def EHE_encoder(**kwargs):
    model = Encoder(
        depths=[2, 2, 2, 2],
        dims=[32, 64, 128, 256],
        token_mixers=[SepConv, SepConv, EfficientAttention, EfficientAttention],
        position_embeddings=[[None, None], [None, None], [None, None], [None, None]],
        **kwargs)
    return model

# ECBFormer
@register_model
def EoFormer_decoder(**kwargs):
    model = Decoder(
        depths=[2, 2, 2],
        dims=[128, 64, 32, 16],
        token_mixers=[[EoS, EoL], [EoS, EoL], [EoS, EoL]],
        **kwargs)
    return model