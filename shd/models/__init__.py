"""
Copyright © 2025 YuQiang's Group at TJU
Full paper: HetSyn: Versatile Timescale Integration in Spiking Neural Networks via Heterogeneous Synapses
Authors: Zhichao Deng, Zhikun Liu, Junxue Wang, Shengqian Chen, Xiang Wei, Qiang Yu
Corresponding: yuqiang@tju.edu.cn
"""

from .hetsyn_base import HetSynLayerNoSpike, HomNeuLayerNoSpike
from .hetsyn_base import SpikeLayer

# heter-syn output layer (only leaky and no spike)
from .hetsyn_stacked import HetSynOutCell
# heter-syn forward snn layer
from .hetsyn_stacked import HetSynFSNNCell
# heter-syn recurrent snn layer
from .hetsyn_stacked import HetSynRSNNCell

# heter-neuron variants (neuronal tau)
from .hetsyn_stacked import HomNeuOutCell
from .hetsyn_stacked import HomNeuFSNNCell
from .hetsyn_stacked import HomNeuRSNNCell


