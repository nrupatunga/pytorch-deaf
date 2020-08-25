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

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
# from pytorch_lightning import _logger as log
from loguru import logger as log
from torch.nn import functional as F
from torch.utils.data import DataLoader

try:
    from src.dataloader.bsd_loader import BsdDataLoader
    from src.arch.custom_nets import EdgeAwareNet
except Exception:
    log.error('Please run $source settings.sh from root directory')
    sys.exit(1)


class deafLitModel(pl.LightningModule):

    """LightningModule for deep edge aware filters"""

    def __init__(self,
                 data_dir: Union[str, Path],
                 batch_size: int = 64,
                 norm_type: str = 'batchnorm',
                 lr: float = 2e-2,
                 num_workers: int = 6, **kwargs) -> None:
        """ Initialization"""
        super().__init__()

        self._model = EdgeAwareNet()
        self.save_hyperparameters()
        self.dbg = True
        self.eps = 1e-6
        self.w_r = 1e-1
        log.info('---' * 12)
        log.info(f'Parameters: {self.hparams}')
        log.info('---' * 12)

        if self.dbg:
            pass

    def forward(self, x):
        """forward function
        @x: input tensor for foward pass
        """
        return self._model(x)

    def configure_optimizers(self):
        return torch.optim.Adagrad(self.parameters(),
                                   lr=self.hparams.lr,
                                   weight_decay=5e-8,
                                   eps=1e-6)

    def train_dataloader(self):
        """train dataloader"""
        hparams = self.hparams
        d_train = BsdDataLoader(hparams.data_dir,
                                train=True)

        dl_train = DataLoader(d_train,
                              batch_size=hparams.batch_size,
                              shuffle=True,
                              num_workers=hparams.num_workers)

        return dl_train

    def _calculate_loss(self, y_hat, y):
        """Calculate the loss
        """
        # loss_1 = F.mse_loss(y_hat, y, reduction='sum') * 0.5
        loss_1 = F.smooth_l1_loss(y_hat, y, reduction='sum') * 0.5
        loss_2 = torch.sqrt(torch.square(y_hat) + self.eps).sum()
        loss = (loss_1 + self.w_r * loss_2) / y_hat.shape[0]

        return loss

    def _vis_images(self, y, idx, prefix='val'):
        y_hat_dbg = y.detach().clone()
        y_hat_dbg = y_hat_dbg[0:10]
        dbg_imgs = y_hat_dbg.cpu().numpy()
        for i in range(dbg_imgs.shape[0]):
            img = dbg_imgs[i]
            img = np.transpose(img, [1, 2, 0])
            img = cv2.normalize(img, None, alpha=0, beta=1,
                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            dbg_imgs[i] = np.transpose(img, [2, 0, 1])

        dbg_imgs = torch.tensor(dbg_imgs)

        grid = torchvision.utils.make_grid(dbg_imgs)
        if idx == 0:
            self.logger.experiment.add_image(f'{prefix}_gt', grid,
                                             self.global_step)
        elif idx == 1:
            self.logger.experiment.add_image(f'{prefix}_preds', grid,
                                             self.global_step)
        else:
            self.logger.experiment.add_image(f'{prefix}_input', grid,
                                             self.global_step)

    def training_step(self, batch, batch_idx):
        """Single node training step"""
        x, y = batch
        x, y = x.float(), y.float()
        y_hat = self(x)
        loss = self._calculate_loss(y_hat, y)
        if batch_idx % 250 == 0:
            self._vis_images(y, 0, prefix='train')
            self._vis_images(y_hat, 1, prefix='train')
            self._vis_images(x, 2, prefix='train')

        tf_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tf_logs}

    def val_dataloader(self):
        """validation dataloader
        """
        hparams = self.hparams
        d_val = BsdDataLoader(hparams.data_dir,
                              train=False)

        dl_val = DataLoader(d_val,
                            batch_size=hparams.batch_size,
                            shuffle=False,
                            num_workers=hparams.num_workers)

        return dl_val

    def validation_step(self, batch, batch_idx):
        """validation step
        @batch: current batch
        @batch_idx: batch index
        """
        x, y = batch
        x, y = x.float(), y.float()
        y_hat = self(x)
        loss = self._calculate_loss(y_hat, y)
        if batch_idx % 200 == 0:
            self._vis_images(y, 0)
            self._vis_images(y_hat, 1)
            self._vis_images(x, 2)

        tf_logs = {'val_loss': loss}
        return {'loss': loss, 'log': tf_logs}

    def validation_epoch_end(self, outputs):
        """end of validation
        """
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}
