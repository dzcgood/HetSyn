"""
    Adopted from Yin, Bojian, Federico Corradi, and Sander M. Bohté. “Accurate and Efficient Time-Domain Classification
    with Adaptive Spiking Recurrent Neural Networks.” Nature Machine Intelligence 3, no. 10 (October 2021): 905–13.
    https://doi.org/10.1038/s42256-021-00397-w.
"""
import os
import tables
import numpy as np
from pathlib import Path

from tqdm import tqdm



data_path_root = Path('./data/SHD')
files = [str(data_path_root / 'test.h5'), str(data_path_root / 'train.h5')]


def binary_image_readout(times, units, dt=1e-3, time_scale=1.0):
    if time_scale != 1.0:
        times = times * time_scale
    img = []
    N = int(1 * time_scale / dt)
    for i in range(N):
        idxs = np.argwhere(times <= i * dt).flatten()
        vals = units[idxs]
        vals = vals[vals > 0]
        vector = np.zeros(700)
        vector[700 - vals] = 1
        times = np.delete(times, idxs)
        units = np.delete(units, idxs)
        img.append(vector)
    return np.array(img)


def generate_dataset(file_name, output_dir, dt=1e-3, time_scale=1.0):
    fileh = tables.open_file(file_name, mode='r')
    units = fileh.root.spikes.units
    times = fileh.root.spikes.times
    labels = fileh.root.labels
    os.mkdir(output_dir)
    # This is how we access spikes and labels
    print("Number of samples: ", len(times))
    for i in tqdm(range(len(times))):
        x_tmp = binary_image_readout(times[i], units[i], dt=dt, time_scale=time_scale)
        y_tmp = labels[i]
        output_file_name = Path(output_dir) / f'ID_{i}_{y_tmp}.npy'
        np.save(output_file_name, x_tmp)
    print('done..')
    return 0


generate_dataset(files[0], output_dir=data_path_root / 'test_4ms', dt=4e-3, time_scale=1.0)
generate_dataset(files[1], output_dir=data_path_root / 'train_4ms', dt=4e-3, time_scale=1.0)


