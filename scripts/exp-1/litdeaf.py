"""
File: litdeaf.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: pytorch lightning trainer
"""

import sys
from pathlib import Path
from typing import Union

import pytorch_lightning as pl
import torch
from pytorch_lightning import _logger as log
from torch.nn import functional as F
from torch.utils.data import DataLoader

try:
    from dataloader.bsd_loader import BsdDataLoader
    from arch.custom_nets import EdgeAwareNet
except Exception:
    log.error('Please run $source settings.sh from root directory')
    sys.exit(1)


class deafLitModel(pl.LightningModule):

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
        self._model = EdgeAwareNet()

    def forward(self, x):
        """forward function
        @x: input tensor for foward pass
        """
        return self._model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        """train dataloader"""
        d_train = BsdDataLoader(self._data_dir,
                                train=True)

        dl_train = DataLoader(d_train,
                              batch_size=self._batch_size,
                              shuffle=True,
                              num_workers=self._num_workers)

        return dl_train

    def training_step(self, batch, batch_idx):
        """Single node training step"""
        x, y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)

        tf_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tf_logs}
