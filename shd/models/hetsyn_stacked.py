"""
Copyright © 2025 YuQiang's Group at TJU
Full paper: HetSyn: Versatile Timescale Integration in Spiking Neural Networks via Heterogeneous Synapses
Authors: Zhichao Deng, Zhikun Liu, Junxue Wang, Shengqian Chen, Xiang Wei, Qiang Yu
Corresponding: yuqiang@tju.edu.cn
"""

import math
import torch
import torch.nn as nn

from .hetsyn_base import HetSynLayerNoSpike, HomNeuLayerNoSpike
from .hetsyn_base import SpikeLayer


class HetSynFSNNCell(nn.Module):
    r"""Feedforward layer with heter-syn.
    """
    def __init__(self,
                 n_in, n_out,
                 learning_w=True,
                 learning_rho=True,
                 bias=False,
                 rho_l=0.0,
                 rho_h=1.0,
                 dt=1.0e-3,
                 thr=1.0,
                 tau_r=20e-3,
                 sg_scale=0.3,
                 sg_width=1.0,
                 rng=None,
                 device=None, dtype=torch.float):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.n_in = n_in
        self.n_out = n_out
        self.learning_w = learning_w
        self.learning_rho = learning_rho  # learning synaptic tau
        self.bias = bias
        self.rho_l = rho_l  # bound for running effective rho
        self.rho_h = rho_h
        self.dt = dt

        self.thr = thr  # firing thr
        self.tau_r = tau_r  # reset current tau
        self.sg_scale = sg_scale   # surrogate scale
        self.sg_width = sg_width   # surrogate width (rel. half)

        if rng is None:
            self.rng = torch.Generator(device=self.device)
        else:
            self.rng = rng

        self.stack_layers()
        # self.set_spike_fun()
        self.create_alias()

        # initialize parameters (similar init could be done outside, customize)
        self.initialize_params()
        # # need to be called whenever params change
        # self.align_params(force=True)

    def stack_layers(self):
        self.layer_in = HetSynLayerNoSpike(
            n_in=self.n_in, n_out=self.n_out,
            learning_w=self.learning_w, learning_rho=self.learning_rho,
            bias=self.bias, rho_l=self.rho_l, rho_h=self.rho_h,
            dt=self.dt, rng=self.rng, **self.factory_kwargs
        )
        self.layer_spike = SpikeLayer(
            n_out=self.n_out, thr=self.thr, tau_r=self.tau_r, dt=self.dt,
            sg_scale=self.sg_scale, sg_width=self.sg_width,
            rng=self.rng, **self.factory_kwargs
        )

    def create_alias(self):
        # alias parameters
        self.w_in = self.layer_in.w
        self.rho_in = self.layer_in.rho
        self.b_in = self.layer_in.b

    def initialize_params(
        self,
        w_mean=0., w_std=0.01,
        tau_mean=100e-3, tau_std=50e-3,
        rng=None, **kwargs
        ):
        r"""only implement a simple default case.
        Complex or more customized ones could be prepared at net or task file.

        # with torch.no_grad():
        #     initialize Parameters
        """
        if rng is None:
            rng = self.rng

        self.layer_in.initialize_params(
            w_mean=w_mean, w_std=w_std,
            tau_mean=tau_mean, tau_std=tau_std, rng=rng
        )

    def align_params(self, force=False):
        self.layer_in.align_params(force=force)

    def forward(self, x, h=None):
        r"""
        Input:
            x: input spikes,  shape: #bi
            h: (hz, hIsyn, hIr),  shape: (#bj, bji, bj)
        Output:
            (z, Isyn, Ir): (#bj, bji, bj)
        """
        if h is None:
            zeros_j = torch.zeros(x.size(0), self.n_out, **self.factory_kwargs)
            zeros_ji = torch.zeros(x.size(0), self.n_out, self.n_in,
                                   **self.factory_kwargs)
            h = (zeros_j, zeros_ji, zeros_j)
        hz = h[0]
        hIsyn = h[1]
        hIr = h[2]

        v_in, Isyn = self.layer_in(x, hIsyn)
        z, Ir = self.layer_spike(v_in, (hz, hIr))

        return (z, Isyn, Ir)

class HetSynOutCell(HetSynFSNNCell):
    r"""Feedforward hetersyn layer without spiking: act as output layer (leaky)
    """
    def stack_layers(self):
        self.layer_in = HetSynLayerNoSpike(
            n_in=self.n_in, n_out=self.n_out,
            learning_w=self.learning_w, learning_rho=self.learning_rho,
            bias=self.bias, rho_l=self.rho_l, rho_h=self.rho_h,
            dt=self.dt, rng=self.rng, **self.factory_kwargs
        )

    def forward(self, x, h=None):
        r"""
        Input:
            x: input spikes,  shape: #bi
            h: hIsyn,  shape: (bji)
        Output:
            (v, Isyn): (#bj, bji)
        """
        if h is None:
            h = torch.zeros(x.size(0), self.n_out, self.n_in, **self.factory_kwargs)

        hIsyn = h

        v, Isyn = self.layer_in(x, hIsyn)

        return (v, Isyn)

class HetSynRSNNCell(HetSynFSNNCell):
    r"""RSNN hetersyn
    """
    def __init__(self, self_rec=True, **kwargs):

        self.self_rec = self_rec # set True as default
        super().__init__(**kwargs)

    def stack_layers(self):
        self.layer_in = HetSynLayerNoSpike(
            n_in=self.n_in, n_out=self.n_out,
            learning_w=self.learning_w, learning_rho=self.learning_rho,
            bias=self.bias, rho_l=self.rho_l, rho_h=self.rho_h,
            dt=self.dt, rng=self.rng, **self.factory_kwargs
        )

        self.layer_rec = HetSynLayerNoSpike(
            n_in=self.n_out, n_out=self.n_out,
            learning_w=self.learning_w, learning_rho=self.learning_rho,
            bias=self.bias, rho_l=self.rho_l, rho_h=self.rho_h,
            dt=self.dt, rng=self.rng, **self.factory_kwargs
        )

        self.layer_spike = SpikeLayer(
            n_out=self.n_out, thr=self.thr, tau_r=self.tau_r, dt=self.dt,
            sg_scale=self.sg_scale, sg_width=self.sg_width,
            rng=self.rng, **self.factory_kwargs
        )

    def create_alias(self):
        # alias parameters
        self.w_in = self.layer_in.w
        self.rho_in = self.layer_in.rho
        self.b_in = self.layer_in.b

        self.w_rec = self.layer_rec.w
        self.rho_rec = self.layer_rec.rho
        self.b_rec = self.layer_rec.b

    def initialize_params(
        self,
        w_mean=0., w_std=0.01,
        w_mean_rec=0., w_std_rec=0.01,
        tau_mean=100e-3, tau_std=50e-3,
        rng=None, **kwargs
        ):
        r"""only implement a simple default case.
        Complex or more customized ones could be prepared at net or task file.

        # with torch.no_grad():
        #     initialize Parameters
        """
        if rng is None:
            rng = self.rng

        self.layer_in.initialize_params(
            w_mean=w_mean, w_std=w_std,
            tau_mean=tau_mean, tau_std=tau_std, rng=rng
        )
        self.layer_rec.initialize_params(
            w_mean=w_mean_rec, w_std=w_std_rec,
            tau_mean=tau_mean, tau_std=tau_std, rng=rng
        )

        # set diag to zero
        if not self.self_rec:
            self.w_rec.data.fill_diagonal_(0.0)

    def align_params(self, force=False):
        self.layer_in.align_params(force=force)
        self.layer_rec.align_params(force=force)

        # set diag to zero
        if not self.self_rec:
            self.w_rec.data.fill_diagonal_(0.0)

    def forward(self, x, h=None):
        r"""
        Input:
            x: input spikes,  shape: #bi
            h: (hz, hIsyn_in, hIsyn_rec, hIr),  shape: (#bj, bji, bjj, bj)
        Output:
            (z, Isyn_in, Isyn_rec, Ir): (#bj, bji, bjj, bj)
        """
        if h is None:
            zeros_j = torch.zeros(x.size(0), self.n_out, **self.factory_kwargs)
            zeros_ji = torch.zeros(x.size(0), self.n_out, self.n_in,
                                   **self.factory_kwargs)
            zeros_jj = torch.zeros(x.size(0), self.n_out, self.n_out,
                                   **self.factory_kwargs)
            h = (zeros_j, zeros_ji, zeros_jj, zeros_j)
        hz = h[0]
        hIsyn_in = h[1]
        hIsyn_rec = h[2]
        hIr = h[3]

        v_in, Isyn_in = self.layer_in(x, hIsyn_in)
        v_rec, Isyn_rec = self.layer_rec(hz, hIsyn_rec)
        z, Ir= self.layer_spike(v_in+v_rec, (hz, hIr))

        return (z, Isyn_in, Isyn_rec, Ir)

class HomNeuFSNNCell(HetSynFSNNCell):
    r"""Feedforward layer with heter-neuron (taum).
    """
    def stack_layers(self):
        self.layer_in = HomNeuLayerNoSpike(
            n_in=self.n_in, n_out=self.n_out,
            learning_w=self.learning_w, learning_rho=self.learning_rho,
            bias=self.bias, rho_l=self.rho_l, rho_h=self.rho_h,
            dt=self.dt, rng=self.rng, **self.factory_kwargs
        )
        self.layer_spike = SpikeLayer(
            n_out=self.n_out, thr=self.thr, tau_r=self.tau_r, dt=self.dt,
            sg_scale=self.sg_scale, sg_width=self.sg_width,
            rng=self.rng, **self.factory_kwargs
        )

    def forward(self, x, h=None):
        r"""
        Input:
            x: input spikes,  shape: #bi
            h: (hz, hIsyn, hIr),  shape: (#bj, bj, bj)
        Output:
            (z, Isyn, Ir): (#bj, bj, bj)
        """
        if h is None:
            zeros_j = torch.zeros(x.size(0), self.n_out, **self.factory_kwargs)
            h = (zeros_j, zeros_j, zeros_j)
        hz = h[0]
        hIsyn = h[1]
        hIr = h[2]

        v_in, Isyn = self.layer_in(x, hIsyn)
        z, Ir = self.layer_spike(v_in, (hz, hIr))

        return (z, Isyn, Ir)

class HomNeuOutCell(HomNeuFSNNCell):
    def stack_layers(self):
        self.layer_in = HomNeuLayerNoSpike(
            n_in=self.n_in, n_out=self.n_out,
            learning_w=self.learning_w, learning_rho=self.learning_rho,
            bias=self.bias, rho_l=self.rho_l, rho_h=self.rho_h,
            dt=self.dt, rng=self.rng, **self.factory_kwargs
        )

    def forward(self, x, h=None):
        r"""
        Input:
            x: input spikes,  shape: #bi
            h: hIsyn,  shape: (bj)
        Output:
            (v, Isyn): (#bj, bj)
        """
        if h is None:
            h = torch.zeros(x.size(0), self.n_out, **self.factory_kwargs)

        hIsyn = h

        v, Isyn = self.layer_in(x, hIsyn)

        return (v, Isyn)

class HomNeuRSNNCell(HomNeuFSNNCell):

    def __init__(self, self_rec=False, **kwargs):
        self.self_rec = self_rec # exclude self connection
        super().__init__(**kwargs)

    def stack_layers(self):
        self.layer_in = HomNeuLayerNoSpike(
            n_in=self.n_in, n_out=self.n_out,
            learning_w=self.learning_w, learning_rho=self.learning_rho,
            bias=self.bias, rho_l=self.rho_l, rho_h=self.rho_h,
            dt=self.dt, rng=self.rng, **self.factory_kwargs
        )
        self.layer_rec = HomNeuLayerNoSpike(
            n_in=self.n_out, n_out=self.n_out,
            learning_w=self.learning_w, learning_rho=self.learning_rho,
            bias=self.bias, rho_l=self.rho_l, rho_h=self.rho_h,
            dt=self.dt, rng=self.rng, **self.factory_kwargs
        )
        self.layer_spike = SpikeLayer(
            n_out=self.n_out, thr=self.thr, tau_r=self.tau_r, dt=self.dt,
            sg_scale=self.sg_scale, sg_width=self.sg_width,
            rng=self.rng, **self.factory_kwargs
        )

    def create_alias(self):
        # alias parameters
        self.w_in = self.layer_in.w
        self.rho_in = self.layer_in.rho
        self.b_in = self.layer_in.b

        self.w_rec = self.layer_rec.w
        self.rho_rec = self.layer_rec.rho
        self.b_rec = self.layer_rec.b

    def initialize_params(
        self,
        w_mean=0., w_std=0.01,
        w_mean_rec=0., w_std_rec=0.01,
        tau_mean=100e-3, tau_std=50e-3,
        rng=None, **kwargs
        ):
        r"""only implement a simple default case.
        Complex or more customized ones could be prepared at net or task file.

        # with torch.no_grad():
        #     initialize Parameters
        """
        if rng is None:
            rng = self.rng

        self.layer_in.initialize_params(
            w_mean=w_mean, w_std=w_std,
            tau_mean=tau_mean, tau_std=tau_std, rng=rng
        )
        self.layer_rec.initialize_params(
            w_mean=w_mean_rec, w_std=w_std_rec,
            tau_mean=tau_mean, tau_std=tau_std, rng=rng
        )

        # set diag to zero
        if not self.self_rec:
            self.w_rec.data.fill_diagonal_(0.0)

    def align_params(self, force=False):
        self.layer_in.align_params(force=force)
        self.layer_rec.align_params(force=force)

        # set diag to zero
        if not self.self_rec:
            self.w_rec.data.fill_diagonal_(0.0)

    def forward(self, x, h=None):
        r"""
        Input:
            x: input spikes,  shape: #bi
            h: (hz, hIsyn_in, hIsyn_rec, hIr),  shape: (#bj, bj, bj, bj)
        Output:
            (z, Isyn_in, Isyn_rec, Ir): (#bj, bj, bj, bj)
        """
        if h is None:
            zeros_j = torch.zeros(x.size(0), self.n_out, **self.factory_kwargs)
            h = (zeros_j, zeros_j, zeros_j, zeros_j)
        hz = h[0]
        hIsyn_in = h[1]
        hIsyn_rec = h[2]
        hIr = h[3]

        v_in, Isyn_in = self.layer_in(x, hIsyn_in)
        v_rec, Isyn_rec = self.layer_rec(hz, hIsyn_rec)
        z, Ir = self.layer_spike(v_in+v_rec, (hz, hIr))

        return (z, Isyn_in, Isyn_rec, Ir)

