"""

File: trainer_deaf.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: pytorch lightning trainer
"""

import sys
from pathlib import Path
from typing import Union

import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from torch.utils.data import DataLoader

try:
    from dataloader.bsd_loader import BsdDataLoader
except Exception as e:
    log.error('Please run $source settings.sh from root directory')
    sys.exit(1)

class deafTrainer(pl.LightningModule):

    """LightningModule for deep edge aware filters"""

    def __init__(self,
                 data_dir: Union[str, Path],
                 batch_size: int = 64,
                 norm_type: str = 'batchnorm',
                 lr: float = 1e-2,
                 num_workers: int = 6, **kwargs) -> None:
        """ Initialization"""
        super().__init__()

        self._data_dir = data_dir
        self._batch_size = batch_size
        self._num_workers = num_workers

    def forward(self, x):
        """forward function
        @x: input tensor for foward pass
        """
        pass

    def train_dataloader(self):
        """train dataloader"""
        d_train = BsdDataLoader(self._data_dir,
                                train=True)

        dl_train = DataLoader(d_train,
                              batch_size=self._batch_size,
                              shuffle=True,
                              num_workers=self._num_workers)

        return dl_train

    def training_step(self):
        """Single node training step"""
        pass

    def training_step_end(self):
        """combine all the training step
        """
        pass
