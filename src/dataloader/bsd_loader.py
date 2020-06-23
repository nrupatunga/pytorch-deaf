"""
File: bsd_loader.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: BSDS500 pathches dataloader
"""
import sys
import time
from pathlib import Path

import cv2
import h5py
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from tqdm import tqdm

try:
    from misc.vis_utils import Visualizer
except Exception as e:
    logger.error('Please run $source settings.sh from root directory')
    sys.exit(1)


class BsdDataLoader(Dataset):

    """Docstring for BsdDataLoader. """

    def __init__(self,
                 path_to_hdf5s: str,
                 train=True):
        """Init"""
        Dataset.__init__(self)

        self._num_patches = 10000

        self._data_files = []
        hdf5s = Path(path_to_hdf5s).glob('*.hdf5')
        for i, hdf5 in tqdm(enumerate(hdf5s)):
            self._data_files.append(hdf5)

        self._num_hdf5s = len(self._data_files)

    def __len__(self):
        """length of the dataset
        """
        return len(self._data_files) * self._num_patches

    def __getitem__(self, idx):
        """get item at idx"""

        idx_hdf5 = int(idx / self._num_patches)
        assert idx_hdf5 <= self._num_hdf5s, 'Number of hdf5s are less \
            than {}'.format(idx)

        img_idx = idx - (idx_hdf5 * self._num_patches)

        hdf5_file = self._data_files[idx_hdf5]
        with h5py.File(hdf5_file, 'r') as f:
            image = f['images_{}'.format(img_idx)][()]
            gt = f['labels_{}'.format(img_idx)][()]

        image = cv2.normalize(image, None, alpha=0, beta=1,
                              norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        gt = cv2.normalize(gt, None, alpha=0, beta=1,
                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return (image, gt)


if __name__ == "__main__":

    path_to_hdf5s = '/Users/nrupatunga/2020/Q2/dataset/L0/train/'
    bsd = BsdDataLoader(path_to_hdf5s=path_to_hdf5s)
    dataloader = DataLoader(bsd, batch_size=1, shuffle=True, num_workers=0)

    viz = Visualizer()
    for i, (img, gt) in enumerate(dataloader):
        data = make_grid(img, nrow=16, pad_value=0)
        label = make_grid(gt, nrow=16, pad_value=0)
        viz.plot_images_np(data, 'data')
        viz.plot_images_np(label, 'gt')
        time.sleep(2)
