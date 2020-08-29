"""
File: train.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: training
"""
import pytorch_lightning as pl
from absl import app, flags
from pytorch_lightning.callbacks import ModelCheckpoint

from litdeaf import deafLitModel

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'data_dir', '/Users/nrupatunga/2020/Q2/dataset/L0/', 'parent director of the data')
flags.DEFINE_string('save_dir', './our_data', 'save directory')
flags.DEFINE_string('save_prefix', 'deaf_', 'prefix for model')
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('num_workers', 6, 'number of workers')
flags.DEFINE_float('lr', 0.001, 'learning rate')
flags.DEFINE_enum('norm_type', 'batch_norm', ['batch_norm',
                                              'layer_norm',
                                              'instance_norm',
                                              'group_norm'],
                  'type of normalization layer')


def main(_):
    """main function"""

    model = deafLitModel(FLAGS.data_dir,
                         FLAGS.batch_size,
                         FLAGS.norm_type,
                         FLAGS.lr,
                         FLAGS.num_workers)
    model.summarize(mode='full')

    ckpt_cb = ModelCheckpoint(filepath=FLAGS.save_dir,
                              save_top_k=-1,
                              save_weights_only=False,
                              prefix=FLAGS.save_prefix)

    trainer = pl.Trainer(default_root_dir=FLAGS.save_dir,
                         gpus=[0, ],
                         min_epochs=10,
                         checkpoint_callback=ckpt_cb,
                         max_epochs=100,
                         progress_bar_refresh_rate=1,
                         profiler=True)
    trainer.fit(model)


if __name__ == "__main__":
    app.run(main)
