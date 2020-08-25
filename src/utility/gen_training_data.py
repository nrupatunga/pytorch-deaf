"""
File: gen_training_data.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: prepare data for training from any image data you have
"""
from pathlib import Path

import cv2
import h5py
import numpy as np
from absl import app, flags
from imutils import paths
from tqdm import tqdm

from L0_Smoothing import L0Smoothing

FLAGS = flags.FLAGS
flags.DEFINE_string('root_dir',
                    '/home/nthere/2020/pytorch-deaf/data/DIV_superres',
                    'root directory of data in train/val folder')
flags.DEFINE_string('save_dir',
                    '/home/nthere/2020/pytorch-deaf/data/DIV_superres/hdf5',
                    'output where hdf5s are dumped')
flags.DEFINE_integer('num_patches', 10000,
                     'number of patches per iteration')
flags.DEFINE_integer('patch_dim', 64,
                     'patch dimension')


def random_crop(image, crop_width, crop_height):
    """Randomly crop input image and return
    """
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop


def edge_extract(S):
    """Edge detection for input and output
    """
    v = np.diff(S, axis=0)
    last_row = S[0, ...] - S[-1, ...]
    last_row = last_row[np.newaxis, ...]
    v = np.vstack([v, last_row])

    return v


def gen_sample_pair(patch, dbg=False):
    """generate one sample pair"""

    S = L0Smoothing(patch, param_lambda=0.02).run()
    S = np.squeeze(S)
    S = np.clip(S, 0, 1)
    vin = edge_extract(patch)
    vout = edge_extract(S)

    if dbg:
        vin_d = cv2.normalize(vin, None, alpha=0, beta=255,
                              norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        vout_d = cv2.normalize(vout, None, alpha=0, beta=255,
                               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        vin_d = cv2.cvtColor(vin_d, cv2.COLOR_BGR2GRAY)
        vout_d = cv2.cvtColor(vout_d, cv2.COLOR_BGR2GRAY)
        out = np.hstack((vin_d, vout_d))
        cv2.imwrite(f'output.png', out)
        __import__('pdb').set_trace()

    # match the implementation of the dataloader
    vin = np.transpose(vin, axes=(2, 0, 1))
    vout = np.transpose(vout, axes=(2, 0, 1))
    return vin, vout


def main(_):
    num_patches = FLAGS.num_patches
    patch_dim = FLAGS.patch_dim

    root_dir = FLAGS.root_dir
    root_dir = Path(root_dir)

    train_dir = root_dir.joinpath('train')
    val_dir = root_dir.joinpath('val')

    train_images = list(paths.list_images(train_dir))
    val_images = list(paths.list_images(val_dir))

    samples = np.zeros((3, patch_dim, patch_dim, num_patches))
    labels = np.zeros_like(samples)

    print(f'Len of train: {len(train_images)}')
    print(f'Len of val: {len(val_images)}')

    for m in range(101):
        print(f'Extracting patch batch: {m + 1} / {101}')
        for i in tqdm(range(num_patches // 8), desc=f'{m}th file:'):
            if m != 100:
                idx = np.random.randint(0, len(train_images))
                img = cv2.imread(str(train_images[idx])) / 255.
            else:
                idx = np.random.randint(0, len(val_images))
                img = cv2.imread(str(val_images[idx])) / 255.

            patch = random_crop(img, patch_dim, patch_dim)
            patch_hor_flip = np.transpose(patch, axes=(1, 0, 2))

            idx_list = np.arange(8)
            for idx in range(4):
                patch_rotated = np.rot90(patch, k=idx)
                vin, vout = gen_sample_pair(patch_rotated)
                samples[..., idx_list[idx]] = vin
                labels[..., idx_list[idx]] = vout

                patch_rotated = np.rot90(patch_hor_flip, k=idx)
                vin, vout = gen_sample_pair(patch_rotated)
                samples[..., idx_list[idx + 4]] = vin
                labels[..., idx_list[idx + 4]] = vout

        Path(FLAGS.save_dir).mkdir(parents=True, exist_ok=True)
        out_file = f'{FLAGS.save_dir}/{m}.hdf5'
        with h5py.File(out_file, 'w') as f:
            for i in range(num_patches):
                f.create_dataset(f'images_{i}', data=samples[..., i],
                                 compression='gzip')
                f.create_dataset(f'labels_{i}', data=labels[..., i],
                                 compression='gzip')


if __name__ == "__main__":
    app.run(main)
