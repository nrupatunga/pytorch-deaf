"""
File: bsd_patches.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: BSDS500 patches
"""
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

mode = 'train'
mat_root_dir = f'/media/nthere/datasets/DIV_superres/patches/train/'
out_root_dir = f'/home/nthere/2020/pytorch-deaf/data/train/'

read = True

if read:
    root_dir = '/home/nthere/2020/pytorch-deaf/data/DIV_superres/hdf5/train/'
    hdf5_files = Path(root_dir).rglob('*.hdf5')
    images = []
    means = []
    stds = []
    for i, f in tqdm(enumerate(hdf5_files)):
        with h5py.File(f) as fout:
            for j in range(10000):
                image = fout['images_{}'.format(j)][()]
                images.append(image)

        if ((i + 1) % 10) == 0:
            images = np.asarray(images)
            means.append(np.mean(images, 0))
            stds.append(np.std(images, 0))
            del images
            images = []

        if (i == 90):
            break

    means = np.asarray(means)
    stds = np.asarray(stds)
    mean = np.mean(means, 1)
    std = np.std(stds, 1)

else:
    for i, mat_file in tqdm(enumerate(Path(mat_root_dir).glob('*.mat'))):
        out_hdf5 = Path(out_root_dir).joinpath('{}.hdf5'.format(i))
        with h5py.File(mat_file, 'r') as f, h5py.File(out_hdf5, 'w') as fout:
            samples_data = np.asarray(list(f['samples']))
            labels_data = np.asarray(list(f['labels']))
            for i, data in enumerate(samples_data):
                fout.create_dataset('images_{}'.format(i),
                                    data=samples_data[i],
                                    compression='gzip')
                fout.create_dataset('labels_{}'.format(i),
                                    data=labels_data[i],
                                    compression='gzip')
