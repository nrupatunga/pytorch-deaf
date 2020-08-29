"""
File: bsd_loader.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga Description: BSDS500 pathches dataloader """
import sys
from pathlib import Path

import cv2
import h5py
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from tqdm import tqdm

try:
    from src.misc.vis_utils import Visualizer
except Exception:
    logger.error('Please run $source settings.sh from root directory')
    sys.exit(1)


class BsdDataLoader(Dataset):

    GT_W = 56
    GT_H = 56

    """Docstring for BsdDataLoader. """

    def __init__(self,
                 path_to_hdf5s: str,
                 train: bool = True,
                 dbg: bool = False):
        """Init"""
        Dataset.__init__(self)

        logger.info('Preparing dataset')

        self._num_patches = 10000
        self._isTrain = train
        self._dbg = dbg

        if self._isTrain:
            hdf5s_path = Path(path_to_hdf5s).joinpath('train')
        else:
            hdf5s_path = Path(path_to_hdf5s).joinpath('val')

        self._data_files = []
        hdf5s = hdf5s_path.rglob('*.hdf5')
        for hdf5 in tqdm(hdf5s):
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
            try:
                image = f['images_{}'.format(img_idx)][()]
                gt = f['labels_{}'.format(img_idx)][()]
            except KeyError as e:
                logger.error(f'File:{hdf5_file}, Key: {img_idx}')
                raise e
            except OSError as e:
                logger.error(f'File:{hdf5_file}, Key: {img_idx}')
                raise e

        idx_h = (image.shape[1] - self.GT_H) // 2
        idx_w = (image.shape[2] - self.GT_W) // 2

        gt = gt[:, idx_h:idx_h + self.GT_H, idx_w:idx_w + self.GT_W]

        assert (gt.shape[1:]) == (self.GT_H, self.GT_W)

        if self._dbg:
            image = cv2.normalize(image, None, alpha=0, beta=1,
                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            gt = cv2.normalize(gt, None, alpha=0, beta=1,
                               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # This might not be needed
        # mean and std, normalization
        # used from matlab implementation of deaf from the author
        image = image - 8.8276e-7
        image = image / 0.1637
        return (image, gt)


if __name__ == "__main__":

    path_to_hdf5s = '/home/nthere/2020/pytorch-deaf/data/DIV_superres/hdf5'
    bsd = BsdDataLoader(path_to_hdf5s=path_to_hdf5s,
                        train=True,
                        dbg=True)
    dataloader = DataLoader(bsd, batch_size=64, shuffle=True, num_workers=0)

    viz = Visualizer()
    for i, (img, gt) in tqdm(enumerate(dataloader)):
        data = make_grid(img, nrow=16, pad_value=0)
        label = make_grid(gt, nrow=16, pad_value=0)
        viz.plot_images_np(data, 'data')
        viz.plot_images_np(label, 'gt')
