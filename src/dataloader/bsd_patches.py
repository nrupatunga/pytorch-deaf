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

mat_root_dir = '/Users/nrupatunga/2020/Q2/dataset/L0/train'

read = True

if read:
    with h5py.File('/Users/nrupatunga/2020/Q2/dataset/L0/train/0.hdf5', 'r') as fout:
        start = time.time()
        for i in range(50000):
            image = fout['images_{}'.format(i)][()]
        end = time.time()
        print(end - start)
        print(image.shape)
else:
    for i, mat_file in tqdm(enumerate(Path(mat_root_dir).glob('*.mat'))):
        out_hdf5 = Path(mat_root_dir).joinpath('{}.hdf5'.format(i))
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