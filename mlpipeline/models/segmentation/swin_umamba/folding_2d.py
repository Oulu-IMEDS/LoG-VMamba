import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from einops import rearrange


class Unfold(nn.Module):
    def __init__(self, kernel_size=3, mode="full"):
        super().__init__()
        self.kernel_size = kernel_size
        assert mode in ["full", "scaled", "half_scaled"]
        self.mode = mode
        UnfoldOp.mode = mode

        weight = torch.eye(kernel_size**2)
        weight = weight.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weight = nn.Parameter(weight, requires_grad=False)

    def _forward_full(self, x):
        b, c, h, w = x.shape
        padding = (self.kernel_size - 1) // 2
        x = F.conv2d(x.reshape(b*c, 1, h, w), self.weight, stride=1, padding=padding)
        x = rearrange(x, "(b c) s h w -> b (c s) h w", b=b, c=c)
        return x

    def forward(self, x):
        if self.mode == "full":
            out = self._forward_full(x)
        else:
            out = UnfoldOp.apply(x, self.weight)
        return out


class UnfoldOp(autograd.Function):
    mode = "scaled"

    @staticmethod
    def forward(tensor, weight):
        b, c, h, w = tensor.shape
        kernel_size = weight.shape[-1]
        padding = (kernel_size - 1) // 2
        padding = [padding, padding]
        out = F.conv2d(tensor.reshape(b*c, 1, h, w), weight, stride=1, padding=padding)
        out = rearrange(out, "(b c) s h w -> b (c s) h w", b=b, c=c)
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        tensor, weight = inputs
        ctx.save_for_backward(tensor, weight)

    @staticmethod
    def backward(ctx, grad_output):
        tensor, weight = ctx.saved_tensors
        b, c, h, w = tensor.shape
        kernel_size = weight.shape[-1]
        padding = (kernel_size - 1) // 2
        s = kernel_size**2
        grad_input = grad_weight = None

        if UnfoldOp.mode in ["scaled", "half_scaled"]:
            grad_output = rearrange(grad_output, "b (c s) h w -> b c s h w", c=c, s=s)
            go = grad_output.reshape(b*c, s, h, w)
            grad_input = F.conv_transpose2d(go, weight.to(go.dtype), stride=1, padding=padding)
            grad_input = grad_input.reshape(b, c, h, w)
            factor = math.sqrt(s) if UnfoldOp.mode == "half_scaled" else s
            grad_input = grad_input / factor
            raise ValueError("Invalid mode!")

        return grad_input, grad_weight
