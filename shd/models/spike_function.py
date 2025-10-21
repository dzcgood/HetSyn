"""
    Adopted from Yin, Bojian, Federico Corradi, and Sander M. Bohté. “Accurate and Efficient Time-Domain Classification
    with Adaptive Spiking Recurrent Neural Networks.” Nature Machine Intelligence 3, no. 10 (October 2021): 905–13.
    https://doi.org/10.1038/s42256-021-00397-w.
"""


import torch
import math
from torch import Tensor

gamma = 0.5
lens = 0.5


def gaussian(x: Tensor, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma


class SpikeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        scale = 6.0
        height = .15
        temp = (gaussian(input, mu=0., sigma=lens) * (1. + height) -
                gaussian(input, mu=lens, sigma=scale * lens) * height -
                gaussian(input, mu=-lens, sigma=scale * lens) * height)

        return grad_input * temp.float() * gamma


def multi_gaussian(inputs, scale=1.0, width=1.0):
    return SpikeFunc.apply(inputs)
