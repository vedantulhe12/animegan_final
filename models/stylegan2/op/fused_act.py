import torch
from torch import nn
from torch.autograd import Function


class FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        if bias is not None:
            input = input + bias.view(1, -1, *([1] * (input.ndim - 2)))

        ctx.save_for_backward(input)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        out = torch.nn.functional.leaky_relu(input, negative_slope) * scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        negative_slope = ctx.negative_slope
        scale = ctx.scale

        grad_input = grad_output.clone()
        grad_input[input < 0] *= negative_slope
        grad_input *= scale

        dim = [0]
        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        grad_bias = grad_input.sum(dim).detach()

        return grad_input, grad_bias, None, None


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)
