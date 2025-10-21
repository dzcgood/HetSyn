"""
Copyright © 2025 YuQiang's Group at TJU
Full paper: HetSyn: Versatile Timescale Integration in Spiking Neural Networks via Heterogeneous Synapses
Authors: Zhichao Deng, Zhikun Liu, Junxue Wang, Shengqian Chen, Xiang Wei, Qiang Yu
Corresponding: yuqiang@tju.edu.cn
"""

import math
import torch
import torch.nn as nn

# heter-syn variants
from .hetsyn_stacked import HetSynOutCell
from .hetsyn_stacked import HetSynFSNNCell
from .hetsyn_stacked import HetSynRSNNCell

from .hetsyn_stacked import HomNeuOutCell
from .hetsyn_stacked import HomNeuFSNNCell
from .hetsyn_stacked import HomNeuRSNNCell


###############################################################################
# Homogeneous LIF: fixed and homo tau_m
###############################################################################
class HomNeuLIFFSNN(nn.Module):
    r"""homo lif forward snn (typical LIF nets as baseline)
    """
    def __init__(self,
                 n_in=10,
                 n_hid=100,
                 n_out=2,
                 dt=1e-3,
                 self_rec=False,    #@@@@ no self connection default
                 learning_w=True,
                 learning_rho=False,#@@@ fixed
                 bias=False,
                 rho_l=0.0,
                 rho_h=1.0,
                 thr=1.0,
                 tau_r=20e-3,
                 sg_scale=0.3,
                 sg_width=1.0,
                 rng=None,
                 dtype=torch.float,
                 device=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        if rng is None:
            self.rng = torch.Generator(device=self.device)
        else:
            self.rng = rng

        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.dt = dt
        self.self_rec = self_rec

        self.learning_w = learning_w
        self.learning_rho = learning_rho  # learning synaptic tau
        self.bias = bias
        self.rho_l = rho_l  # bound for running effective rho
        self.rho_h = rho_h
        self.thr = thr  # firing thr
        self.tau_r = tau_r  # reset current tau
        self.sg_scale = sg_scale   # surrogate scale
        self.sg_width = sg_width   # surrogate width (rel. half)

        # create layers
        self.create_hidden_layer()
        self.create_output_layer()

        self.create_alias()

    def create_hidden_layer(self):
        self.layer_hidden = HomNeuFSNNCell(
            n_in=self.n_in, n_out=self.n_hid, learning_w=self.learning_w,
            learning_rho=self.learning_rho, #@@@ variants: manually forced
            bias=self.bias,
            rho_l=self.rho_l, rho_h=self.rho_h, dt=self.dt, thr=self.thr,
            tau_r=self.tau_r, sg_scale=self.sg_scale, sg_width=self.sg_width,
            rng=self.rng,
            device=self.device, dtype=self.dtype
        )

    def create_output_layer(self):
        self.layer_out = HomNeuOutCell(
            n_in=self.n_hid, n_out=self.n_out, learning_w=self.learning_w,
            learning_rho=self.learning_rho, #@@@ variants: manually forced
            bias=self.bias,
            rho_l=self.rho_l, rho_h=self.rho_h, dt=self.dt, thr=self.thr,
            tau_r=self.tau_r, sg_scale=self.sg_scale, sg_width=self.sg_width,
            rng=self.rng,
            device=self.device, dtype=self.dtype
        )

    def create_alias(self):
        self.w_in = self.layer_hidden.w_in
        self.w_out = self.layer_out.w_in
        self.b_in = self.layer_hidden.b_in
        self.b_out = self.layer_out.b_in

    def initialize_params(
        self,
        w_mean=0., w_std=1.0,
        tau_mean=50e-3, tau_std=0.0, #@@@ variants: homo init
        rng=None, **kwargs):
        r""" default w init: normal: mean=0.0, std=1.0/sqrt(nin)
        """
        if rng is None:
            rng = self.rng

        self.layer_hidden.initialize_params(
            w_mean=w_mean, w_std=w_std/(self.n_in**0.5),
            tau_mean=tau_mean, tau_std=tau_std, rng=rng
        )
        self.layer_out.initialize_params(
            w_mean=w_mean, w_std=w_std/(self.n_hid**0.5),
            tau_mean=tau_mean, tau_std=tau_std, rng=rng
        )

    def align_params(self):
        self.layer_hidden.align_params()
        self.layer_out.align_params()

    def forward(self, inputs):
        # inputs must be batch_first: b, t, i

        inputs = inputs.transpose(0, 1) # switch b, t to t, b

        time_steps = inputs.shape[0]
        batch_size = inputs.shape[1]

        h = None
        hout = None

        z_outputs = []   # spikes of hidden layer
        mem_outputs = [] # mem potential of the output leaky layer
        v_outputs = []

        for i in range(time_steps):
            xin = inputs[i]
            # h: (hz, hIsyn, hIr)
            h = self.layer_hidden(xin, h)
            z = h[0]
            # v = h[-1]
            z_outputs.append(z)
            # v_outputs.append(v)

            vout, hout = self.layer_out(z, hout)
            mem_outputs.append(vout)

        # batch, time_steps, n_out
        y_out = torch.stack(mem_outputs, dim=1)
        z_hidden = torch.stack(z_outputs, dim=1)
        # v_out = torch.stack(v_outputs,dim=1)

        return (y_out, z_hidden)

class HomNeuLIFRSNN(HomNeuLIFFSNN):
    r"""homo lif recurrent snn (typical LIF nets as baseline)
    """
    def create_hidden_layer(self):
        self.layer_hidden = HomNeuRSNNCell(
            self_rec=self.self_rec,
            n_in=self.n_in, n_out=self.n_hid, learning_w=self.learning_w,
            learning_rho=self.learning_rho, #@@@ variants: manually forced
            bias=self.bias,
            rho_l=self.rho_l, rho_h=self.rho_h, dt=self.dt, thr=self.thr,
            tau_r=self.tau_r, sg_scale=self.sg_scale, sg_width=self.sg_width,
            rng=self.rng,
            device=self.device, dtype=self.dtype
        )

    def create_alias(self):
        self.w_in = self.layer_hidden.w_in
        self.w_rec = self.layer_hidden.w_rec
        self.w_out = self.layer_out.w_in
        self.b_in = self.layer_hidden.b_in
        self.b_rec = self.layer_hidden.b_rec
        self.b_out = self.layer_out.b_in

    def initialize_params(
        self,
        w_mean=0., w_std=1.0,
        w_mean_rec=0., w_std_rec=1.0,
        tau_mean=50e-3, tau_std=0.0, #@@@ variants: homo init
        rng=None, **kwargs):
        r""" default w init: normal: mean=0.0, std=1.0/sqrt(nin)
        """
        if rng is None:
            rng = self.rng

        self.layer_hidden.initialize_params(
            w_mean=w_mean, w_std=w_std/(self.n_in**0.5),
            w_mean_rec=w_mean_rec, w_std_rec=w_std_rec/(self.n_hid**0.5),
            tau_mean=tau_mean, tau_std=tau_std, rng=rng
        )
        self.layer_out.initialize_params(
            w_mean=w_mean, w_std=w_std/(self.n_hid**0.5),
            tau_mean=tau_mean, tau_std=tau_std, rng=rng
        )


###############################################################################
# Heter-Syn (HS)
###############################################################################
class HetSynLIFFSNN(HomNeuLIFFSNN):
    def __init__(self, self_rec=True, learning_rho=True, **kwargs):
        super().__init__(self_rec=self_rec, learning_rho=learning_rho, **kwargs)

    def create_hidden_layer(self):
        self.layer_hidden = HetSynFSNNCell(
            n_in=self.n_in, n_out=self.n_hid, learning_w=self.learning_w,
            learning_rho=self.learning_rho, #@@@ variants: manually forced
            bias=self.bias,
            rho_l=self.rho_l, rho_h=self.rho_h, dt=self.dt, thr=self.thr,
            tau_r=self.tau_r, sg_scale=self.sg_scale, sg_width=self.sg_width,
            rng=self.rng,
            device=self.device, dtype=self.dtype
        )

    def create_output_layer(self):
        self.layer_out = HetSynOutCell(
            n_in=self.n_hid, n_out=self.n_out, learning_w=self.learning_w,
            learning_rho=self.learning_rho, #@@@ variants: manually forced
            bias=self.bias,
            rho_l=self.rho_l, rho_h=self.rho_h, dt=self.dt, thr=self.thr,
            tau_r=self.tau_r, sg_scale=self.sg_scale, sg_width=self.sg_width,
            rng=self.rng,
            device=self.device, dtype=self.dtype
        )

    def create_alias(self):
        self.w_in = self.layer_hidden.w_in
        self.w_out = self.layer_out.w_in
        self.rho_in = self.layer_hidden.rho_in
        self.rho_out = self.layer_out.rho_in
        self.b_in = self.layer_hidden.b_in
        self.b_out = self.layer_out.b_in

class HetSynLIFRSNN(HetSynLIFFSNN):

    def create_hidden_layer(self):
        self.layer_hidden = HetSynRSNNCell(
            self_rec=self.self_rec,
            n_in=self.n_in, n_out=self.n_hid, learning_w=self.learning_w,
            learning_rho=self.learning_rho, #@@@ variants: manually forced
            bias=self.bias,
            rho_l=self.rho_l, rho_h=self.rho_h, dt=self.dt, thr=self.thr,
            tau_r=self.tau_r, sg_scale=self.sg_scale, sg_width=self.sg_width,
            rng=self.rng,
            device=self.device, dtype=self.dtype
        )

    def create_alias(self):
        super().create_alias()
        self.w_rec = self.layer_hidden.w_rec
        self.rho_rec = self.layer_hidden.rho_rec
        self.b_rec = self.layer_hidden.b_rec

    def initialize_params(
        self,
        w_mean=0., w_std=1.0,
        w_mean_rec=0., w_std_rec=1.0,
        tau_mean=100e-3, tau_std=50e-3, #@@@ variants: heter init
        rng=None, **kwargs):
        r""" default w init: normal: mean=0.0, std=1.0/sqrt(nin)
        """
        if rng is None:
            rng = self.rng

        self.layer_hidden.initialize_params(
            w_mean=w_mean, w_std=w_std/(self.n_in**0.5),
            w_mean_rec=w_mean_rec, w_std_rec=w_std_rec/(self.n_hid**0.5),
            tau_mean=tau_mean, tau_std=tau_std, rng=rng
        )
        self.layer_out.initialize_params(
            w_mean=w_mean, w_std=w_std/(self.n_hid**0.5),
            tau_mean=tau_mean, tau_std=tau_std, rng=rng
        )
