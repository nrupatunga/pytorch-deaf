"""
File: bsd_patches.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: BSDS500 patches
"""
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

mat_root_dir = '/media/nthere/datasets/deaf/L0/train/'

with h5py.File('/media/nthere/datasets/deaf/L0/train/0.hdf5', 'r') as fout:
    start = time.time()
    image = fout['samples'][()][0]
    end = time.time()
    print(end - start)
    print(image.shape)

# for i, mat_file in tqdm(enumerate(Path(mat_root_dir).glob('*.mat'))):
    # out_hdf5 = Path(mat_root_dir).joinpath('{}.hdf5'.format(i))
    # with h5py.File(mat_file, 'r') as f, h5py.File(out_hdf5, 'w') as fout:
        # samples_data = np.asarray(list(f['samples']))
        # labels_data = np.asarray(list(f['labels']))
        # fout.create_dataset('samples', data=samples_data,
                            # compression='gzip')
        # fout.create_dataset('labels', data=labels_data,
                            # compression='gzip')

