"""
File: train.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: training
"""
import pytorch_lightning as pl
from absl import app, flags
from pytorch_lightning import _logger as log

from trainer_deaf import deafTrainer

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', './', 'parent director of the data')
flags.DEFINE_integer('batch_size', 64, 'batch size')
flags.DEFINE_integer('num_workers', 6, 'number of workers')
flags.DEFINE_float('lr', 0.01, 'learning rate')
flags.DEFINE_enum('norm_type', 'batch_norm', ['batch_norm',
                                              'layer_norm',
                                              'instance_norm',
                                              'group_norm'],
                  'type of normalization layer')


def main(args):
    """main function"""

    trainer = deafTrainer(FLAGS.data_dir,
                          FLAGS.batch_size,
                          FLAGS.norm_type,
                          FLAGS.lr,
                          FLAGS.num_workers)
    return trainer


if __name__ == "__main__":
    app.run(main)
