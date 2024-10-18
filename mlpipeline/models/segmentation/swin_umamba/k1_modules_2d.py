import time
import math
import numpy as np
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from mlpipeline.models.segmentation.swin_umamba.folding_2d import Unfold


class SS2D_K1(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True) # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 1

        xs = x.contiguous().view(B, 1, -1, L)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        return out_y[:, 0]

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y = self.forward_core(x)
        assert y.dtype == torch.float32
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)

        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class SS2Depth_K1(nn.Module):
    def __init__(
        self,
        d_model,
        mode,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        d_depth_stride=2,
        d_depth_out=1,
        d_depth_squeeze=1,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_depth_squeeze = d_depth_squeeze

        self.mode = mode
        assert mode in ["head", "center", "split", "even"]
        print("K1")
        print("depth:", self.mode, d_depth_squeeze, d_depth_out, d_depth_stride)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.depth_conv = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner // d_depth_squeeze,
            groups=self.d_inner // d_depth_squeeze,
            bias=conv_bias,
            kernel_size=d_conv + d_depth_stride - 2,
            # kernel_size=d_conv,
            # dilation=d_depth_stride // 2,
            padding=((d_conv + d_depth_stride - 3) // 2),
            stride=d_depth_stride,
        )
        self.depth_fc = nn.Linear(d_depth_out, self.d_inner)
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        # (K=4, N, inner)
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        # (K=4, inner, rank)
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        # (K=4, D, N)
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, dx: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        CC = C // self.d_depth_squeeze
        LL = L + CC
        K = 1
        half_length = L // 2

        xs = x.contiguous().view(B, 1, -1, L)
        dx = torch.unsqueeze(dx, dim=1)
        interval = int(math.ceil(xs.shape[-1] / dx.shape[-1]))
        # print(xs.shape, H * W, dx.shape)
        if self.mode == "head":
            xs = torch.cat([dx, xs], dim=-1)
        elif self.mode == "center":
            xs = torch.cat([xs[..., :half_length], dx, xs[..., half_length:]], dim=-1)
        elif self.mode == "split":
            dx_half_length = dx.shape[-1] // 2
            xs = torch.cat([
                dx[..., :dx_half_length],
                xs[..., :half_length],
                dx[..., dx_half_length:],
                xs[..., half_length:],
            ], dim=-1)
        elif self.mode == "even":
            remaining_length_of_dx = dx.shape[-1] - (xs.shape[-1] // interval)
            new_xs = torch.empty([xs.shape[0], xs.shape[1], xs.shape[2], LL], dtype=xs.dtype, device=xs.device)
            start_dx_index = int(math.ceil(remaining_length_of_dx / (interval + 1))) * (interval + 1)
            dx_indices = torch.arange(start_dx_index, LL, interval + 1).to(xs.device)
            dx_indices = torch.tile(dx_indices, [xs.shape[0], xs.shape[1], xs.shape[2], 1])
            remaining_length_of_dx = dx.shape[-1] - dx_indices.shape[-1]
            new_xs.scatter_(dim=-1, index=dx_indices, src=dx[..., remaining_length_of_dx:])
            indices = torch.arange(remaining_length_of_dx, LL).to(xs.device)
            indices = indices[(indices % (interval + 1) != 0)]
            indices = torch.tile(indices, [xs.shape[0], xs.shape[1], xs.shape[2], 1])
            new_xs.scatter_(dim=-1, index=indices, src=xs)
            new_xs[..., :remaining_length_of_dx] = dx[..., :remaining_length_of_dx]
            # print(dx_indices.shape, dx.shape, remaining_length_of_dx, indices.shape, new_xs.mean().item())
            xs = new_xs
        # print(xs.shape, self.bidirectional, interval)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, LL), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, LL), self.dt_projs_weight)

        xs = xs.float().view(B, -1, LL) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, LL) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, LL) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, LL) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, LL)
        if self.mode == "head":
            out_y = out_y[..., CC:]
        elif self.mode in ["split"]:
            out_y = torch.cat([
                out_y[..., dx_half_length:(dx_half_length + half_length)],
                out_y[..., -half_length:],
            ], dim=-1)
        elif self.mode in ["center"]:
            out_y = torch.cat([
                out_y[..., :half_length],
                out_y[..., -half_length:],
            ], dim=-1)
        elif self.mode == "even":
            out_y = torch.gather(out_y, dim=-1, index=indices)
            # print(out_y.mean().item())
            assert out_y.shape[-1] == L
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        dx = self.depth_conv(x)
        # print(x.shape, dx.shape)
        dx = rearrange(dx, "b c h w -> b c (h w)")
        dx = self.depth_fc(dx).permute(0, 2, 1)
        dx = self.act(dx)

        y = self.forward_core(x, dx)
        assert y.dtype == torch.float32
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)

        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class SS2DConv_K1(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        squeeze=8,
        no_squeeze=True,
        neighbors=9,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model // squeeze * neighbors
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * d_model)
        d_fold_out = self.d_inner // squeeze * neighbors
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        d_sequence_model = d_fold_out
        copies = 1

        d_conv_input = self.d_inner // squeeze if no_squeeze else self.d_inner
        self.in_proj = nn.Linear(d_model, d_conv_input * 2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=d_conv_input,
            out_channels=self.d_inner // squeeze,
            groups=self.d_inner // squeeze,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.unfold = Unfold(d_conv)
        self.fold = nn.Linear(d_sequence_model, d_conv_input, bias=bias)

        print("K1")
        print("new size:", self.dt_rank, self.d_model, d_sequence_model, d_model)
        self.x_proj = (
            nn.Linear(d_sequence_model, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        # (K=4, N, inner)
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, d_sequence_model, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        # (K=4, N, inner)
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        # (K=4, D, N)
        self.A_logs = self.A_log_init(self.d_state, d_sequence_model, copies=copies, merge=True)
        self.Ds = self.D_init(d_sequence_model, copies=copies, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(d_conv_input)
        self.out_proj = nn.Linear(d_conv_input, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 1

        xs = x.contiguous().view(B, 1, -1, L)
        # print("xs:", xs.min().item(), xs.max().item())

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, **kwargs):
        # self.A_logs = self.A_log_init(self.d_state, self.d_inner_proj, copies=4, merge=True)
        # self = self.cuda()
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        # (b, d, h, w)
        x = self.act(self.conv2d(x))
        with torch.enable_grad():
            x = self.unfold(x)

        y = self.forward_core(x)
        assert y.dtype == torch.float32
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.fold(y)
        y = self.out_norm(y)
        y = y * F.silu(z)

        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class SS2DepthConv_K1(nn.Module):
    def __init__(
        self,
        d_model,
        mode,
        d_state=16,
        d_conv=3,
        squeeze=8,
        neighbors=9,
        d_depth_stride=2,
        d_depth_out=1,
        d_depth_squeeze=1,
        conv_mode="",
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model // squeeze * neighbors
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * d_model)
        d_fold_out = self.d_inner // squeeze * neighbors
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_depth_squeeze = d_depth_squeeze
        d_sequence_model = d_fold_out

        self.mode = mode
        assert mode in ["head", "center", "split", "even"]
        print("K1")
        print("depth:", self.mode, d_depth_squeeze, d_depth_out, d_depth_stride)

        d_conv_input = self.d_inner
        # d_conv_input = self.d_inner
        self.in_proj = nn.Linear(d_model, d_conv_input * 2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=d_conv_input,
            out_channels=self.d_inner // squeeze,
            groups=self.d_inner // squeeze,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.depth_conv = nn.Conv2d(
            in_channels=d_sequence_model,
            out_channels=d_sequence_model // d_depth_squeeze,
            groups=d_sequence_model // d_depth_squeeze,
            bias=conv_bias,
            kernel_size=d_conv + d_depth_stride - 2,
            padding=((d_conv + d_depth_stride - 3) // 2),
            stride=d_depth_stride,
        )
        # print(d_conv + d_depth_stride - 2, d_depth_stride, ((d_conv + d_depth_stride - 3) // 2))
        self.depth_fc = nn.Linear(d_depth_out, d_sequence_model)
        self.act = nn.SiLU()
        self.unfold = Unfold(d_conv, mode=conv_mode)
        self.fold = nn.Linear(d_sequence_model, d_conv_input, bias=bias)

        print("new size:", self.dt_rank, self.d_model, d_sequence_model, d_model)
        self.x_proj = (
            nn.Linear(d_sequence_model, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        # (K=4, N, inner)
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, d_sequence_model, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        # (K=4, inner, rank)
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        # (K=4, D, N)
        self.A_logs = self.A_log_init(self.d_state, d_sequence_model, copies=1, merge=True)
        self.Ds = self.D_init(d_sequence_model, copies=1, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(d_conv_input)
        self.out_proj = nn.Linear(d_conv_input, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, dx: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        CC = C // self.d_depth_squeeze
        LL = L + CC
        K = 1
        half_length = L // 2

        xs = x.contiguous().view(B, 1, -1, L)
        dx = torch.unsqueeze(dx, dim=1)
        interval = int(math.ceil(xs.shape[-1] / dx.shape[-1]))
        # print(xs.shape, H * W, dx.shape)
        if self.mode == "head":
            xs = torch.cat([dx, xs], dim=-1)
        elif self.mode == "center":
            xs = torch.cat([xs[..., :half_length], dx, xs[..., half_length:]], dim=-1)
        elif self.mode == "split":
            dx_half_length = dx.shape[-1] // 2
            xs = torch.cat([
                dx[..., :dx_half_length],
                xs[..., :half_length],
                dx[..., dx_half_length:],
                xs[..., half_length:],
            ], dim=-1)
        elif self.mode == "even":
            remaining_length_of_dx = dx.shape[-1] - (xs.shape[-1] // interval)
            new_xs = torch.empty([xs.shape[0], xs.shape[1], xs.shape[2], LL], dtype=xs.dtype, device=xs.device)
            start_dx_index = int(math.ceil(remaining_length_of_dx / (interval + 1))) * (interval + 1)
            dx_indices = torch.arange(start_dx_index, LL, interval + 1).to(xs.device)
            dx_indices = torch.tile(dx_indices, [xs.shape[0], xs.shape[1], xs.shape[2], 1])
            remaining_length_of_dx = dx.shape[-1] - dx_indices.shape[-1]
            new_xs.scatter_(dim=-1, index=dx_indices, src=dx[..., remaining_length_of_dx:])
            indices = torch.arange(remaining_length_of_dx, LL).to(xs.device)
            indices = indices[(indices % (interval + 1) != 0)]
            indices = torch.tile(indices, [xs.shape[0], xs.shape[1], xs.shape[2], 1])
            new_xs.scatter_(dim=-1, index=indices, src=xs)
            new_xs[..., :remaining_length_of_dx] = dx[..., :remaining_length_of_dx]
            # print(dx_indices.shape, dx.shape, remaining_length_of_dx, indices.shape, new_xs.mean().item())
            xs = new_xs
        # print(xs.shape, self.bidirectional, interval)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, LL), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, LL), self.dt_projs_weight)

        xs = xs.float().view(B, -1, LL) # (b, k * d, l)
        # print(torch.mean(xs[0, :C, 2]), torch.mean(xs[0, 3*C:4*C, 2]), torch.mean(xs[0, :C, C]), torch.mean(xs[0, 3*C:4*C, C]))
        dts = dts.contiguous().float().view(B, -1, LL) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, LL) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, LL) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, LL)
        if self.mode == "head":
            out_y = out_y[..., CC:]
        elif self.mode in ["split"]:
            out_y = torch.cat([
                out_y[..., dx_half_length:(dx_half_length + half_length)],
                out_y[..., -half_length:],
            ], dim=-1)
        elif self.mode in ["center"]:
            out_y = torch.cat([
                out_y[..., :half_length],
                out_y[..., -half_length:],
            ], dim=-1)
        elif self.mode == "even":
            out_y = torch.gather(out_y, dim=-1, index=indices)
            # print(out_y.mean().item())
            assert out_y.shape[-1] == L
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        x = self.unfold(x)

        dx = self.depth_conv(x)
        # print(x.shape, dx.shape)
        dx = rearrange(dx, "b c h w -> b c (h w)")
        dx = self.depth_fc(dx).permute(0, 2, 1)
        dx = self.act(dx)

        y = self.forward_core(x, dx)
        assert y.dtype == torch.float32
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.fold(y)
        y = self.out_norm(y)
        y = y * F.silu(z)

        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
