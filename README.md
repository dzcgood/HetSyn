# HetSyn: Versatile Timescale Integration in Spiking Neural Networks via Heterogeneous Synapses

This repository provides example code and configurations to reproduce the SHD task from ***HetSyn: Versatile Timescale Integration in Spiking Neural Networks via Heterogeneous Synapses***. 

## Dataset

+ `SHD`: the Spiking Heidelberg dataset can be downloaded from the [official site](https://compneuro.net/datasets/).

## Requirements

The following Python packages are required:

+ `Python` ≥ 3.9
+ `HDF5` for python 
+ `torch` ≥ 2.2.0
+ packages for preprocessing: `tables`, `librosa` and `tqdm`
+ `matplotlib`

## Quick start

The data used to generate the figures in the manuscript can be obtained by adjusting the command-line parameters and executing the corresponding task scripts. 



**Running Experiments on *SHD***

Before training, the SHD dataset should be preprocessed. Navigate to the `shd` directory and run:

```bash
# Preprocess SHD dataset
python shd_generate_dataset.py
```

To train the HetSynLIF model on SHD, navigate to the `SHD` directory and run:

```bash
# HetSynLIF on SHD
python SHD_HetSynLIF_RSNN.py
```

A pre-trained model is also provided for this task in the folder.

## Reference

If you find this work or code useful, please kindly consider citing our paper:

```
@article{deng2025hetsyn,
  title={HetSyn: Versatile Timescale Integration in Spiking Neural Networks via Heterogeneous Synapses},
  author={Deng, Zhichao and Liu, Zhikun and Wang, Junxue and Chen, Shengqian and Wei, Xiang and Yu, Qiang},
  journal={arXiv preprint arXiv:2508.11644},
  year={2025}
}
```

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License. To view a copy of this license, visit [https://creativecommons.org/licenses/by-nc/4.0/](https://creativecommons.org/licenses/by-nc/4.0/).