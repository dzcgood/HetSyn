"""
Copyright © 2025 YuQiang's Group at TJU
Full paper: HetSyn: Versatile Timescale Integration in Spiking Neural Networks via Heterogeneous Synapses
Authors: Zhichao Deng, Zhikun Liu, Junxue Wang, Shengqian Chen, Xiang Wei, Qiang Yu
Corresponding: yuqiang@tju.edu.cn
"""

import math
import torch
import torch.nn as nn

from .spike_function import multi_gaussian


class HetSynLayerNoSpike(nn.Module):
    r"""Feedforward layer with heter syn (without spike output and reset)
    """
    def __init__(self,
                 n_in, n_out,
                 learning_w=True,
                 learning_rho=True,
                 bias=False,
                 rho_l=0.0,
                 rho_h=1.0,
                 dt=1.0e-3,
                 rng=None,
                 device=None, dtype=torch.float):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.n_in = n_in
        self.n_out = n_out
        self.learning_w = learning_w
        self.learning_rho = learning_rho
        self.bias = bias
        self.rho_l = rho_l  # bound for running effective rho
        self.rho_h = rho_h
        self.dt = dt
        if rng is None:
            self.rng = torch.Generator(device=self.device)
        else:
            self.rng = rng

        self.create_params()
        self.initialize_params()

    def create_params(self):
        self.create_w()
        self.create_rho()
        self.create_b()

    def create_w(self):
        # instantiate w with shape of (out, in)
        self.w = nn.Parameter(
            torch.empty((self.n_out, self.n_in), **self.factory_kwargs),
            requires_grad=self.learning_w)

    def create_b(self):
        self.b = nn.Parameter(
            torch.zeros((self.n_out, self.n_in), **self.factory_kwargs),
            requires_grad=self.bias)

    def create_rho(self):
        self.rho = nn.Parameter(
            torch.empty((self.n_out, self.n_in), **self.factory_kwargs),
            requires_grad=self.learning_rho)

    def initialize_params(self,
                          w_mean=0., w_std=0.01,
                          tau_mean=20e-3, tau_std=0.0,
                          rng=None):
        """
        Default is Gaussian with 0. mean and std;
        Otherwise could be initialized outside with torch.no_grad().
        """
        if rng is None:
            rng = self.rng

        # nn.init does not change param grad
        nn.init.normal_(self.w, mean=w_mean, std=w_std, generator=rng)

        tau = torch.normal(mean=tau_mean, std=tau_std,
                           size=(self.n_out, self.n_in),
                           device=self.device, generator=rng)
        self.rho.data.copy_(torch.exp(-self.dt/tau)) # compute rho

        with torch.no_grad():
            self.clamp_params(force=True)

    def clamp_rho(self, force=False):
        if self.learning_rho or force:
            self.rho.data.clamp_(min=self.rho_l, max=self.rho_h)  # track grad

    def clamp_params(self, force=False):
        # should be called whenever params are changed (e.g. init or learn)
        self.clamp_rho(force=force)

    def align_params(self, force=False):
        self.clamp_params(force=force)

    def forward(self, x, h=None):
        r"""
        Input:
            x: input of t,  shape: bi
            h: hI of t-1,  shape: bji
        Output:
            v: bj
        """

        if h is None:
            h = torch.zeros(x.size(0), self.n_out, self.n_in,
                            **self.factory_kwargs)

        hI = h

        # compute I
        I = (hI*self.rho.unsqueeze(0) + self.w.unsqueeze(0)*x.unsqueeze(1)
             + self.b.unsqueeze(0)) # bji
        v = torch.sum(I, dim=2) # bj
        return (v, I)

class HomNeuLayerNoSpike(HetSynLayerNoSpike):
    r"""Feedforward layer with heter neuron (without spike output and reset)
    """
    def create_rho(self):
        self.rho = nn.Parameter(
            torch.empty(self.n_out, **self.factory_kwargs),
            requires_grad=self.learning_rho)

    def create_b(self):
        self.b = nn.Parameter(
            torch.zeros(self.n_out, **self.factory_kwargs),
            requires_grad=self.bias)

    def initialize_params(self,
                          w_mean=0., w_std=0.01,
                          tau_mean=20e-3, tau_std=0.0,
                          rng=None):
        """
        Default is Gaussian with 0. mean and std;
        Otherwise could be initialized outside with torch.no_grad().
        """
        if rng is None:
            rng = self.rng

        # nn.init does not change param grad
        nn.init.normal_(self.w, mean=w_mean, std=w_std, generator=rng)

        tau = torch.normal(mean=tau_mean, std=tau_std,
                           size=(self.n_out,),device=self.device, generator=rng)
        self.rho.data.copy_(torch.exp(-self.dt/tau)) # compute rho

        with torch.no_grad():
            self.clamp_params(force=True)

    def forward(self, x, h=None):
        r"""
        Input:
            x: input of t,  shape: bi
            h: hI of t-1,  shape: bj
        Output:
            v: bj
        """

        if h is None:
            h = torch.zeros(x.size(0), self.n_out, **self.factory_kwargs)

        hI = h

        # compute I (bj)
        I = hI*self.rho.unsqueeze(0) + x @ self.w.T + self.b.unsqueeze(0)
        v = I
        return (v, I)

class SpikeLayer(nn.Module):
    r"""spiking layer with hidden states of reset current.
    """
    def __init__(self,
                 n_out,
                 thr=1.0,
                 tau_r=20e-3,
                 dt=1.0e-3,
                 sg_scale=0.3,
                 sg_width=1.0,
                 rng=None,
                 device=None, dtype=torch.float):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.n_out = n_out
        self.thr = thr  # firing thr
        self.tau_r = tau_r  # reset current tau
        self.dt = dt
        self.rho_r = math.exp(-self.dt/self.tau_r)

        if rng is None:
            self.rng = torch.Generator(device=self.device)
        else:
            self.rng = rng

        self.spike_fun = multi_gaussian # spike function
        self.sg_scale = sg_scale   # surrogate scale
        self.sg_width = sg_width   # surrogate width (rel. half)

    def forward(self, x, h=None):
        r"""
        Input:
            x: Isum,  shape: bj
            h: (hz, Ireset),   shape: bj
        Output:
            (z, Ireset): bj
        """
        if h is None:
            zeros = torch.zeros(x.size(0), self.n_out, **self.factory_kwargs)
            h = (zeros, zeros)

        hz, hIreset = h[0], h[1]

        Ireset = self.rho_r*hIreset + self.thr*hz
        v = x - Ireset
        z = self.spike_fun(v, thr=self.thr, scale=self.sg_scale,
                           width=self.sg_width)

        return (z, Ireset)


