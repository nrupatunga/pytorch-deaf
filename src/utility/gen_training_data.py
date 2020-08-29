"""

File: gen_training_data.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: prepare data for training from any image data you have
"""
import multiprocessing
import random
from multiprocessing import Manager
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

    x = random.randint(0, max_x - 1)
    y = random.randint(0, max_y - 1)

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

    # match the implementation of the dataloader
    vin = np.transpose(vin, axes=(2, 0, 1))
    vout = np.transpose(vout, axes=(2, 0, 1))
    return vin, vout


def generate(params):
    """generate images """
    train_images, val_images, samples_b, labels_b, patch_dim, num_patches, m = params

    samples = samples_b[m]
    labels = labels_b[m]

    for i in range(num_patches // 8):
        print(f'Batch: {m}, Id: {i}')
        if m != 100:
            idx = random.randint(0, len(train_images) - 1)
            img = cv2.imread(str(train_images[idx])) / 255.
            img = img.astype(np.float32)
        else:
            idx = random.randint(0, len(val_images) - 1)
            img = cv2.imread(str(val_images[idx])) / 255.
            img = img.astype(np.float32)

        patch = random_crop(img, patch_dim, patch_dim)
        patch_hor_flip = np.transpose(patch, axes=(1, 0, 2))

        idx_list = np.arange(8)
        for idx in range(4):
            patch_rotated = np.rot90(patch, k=idx)
            vin, vout = gen_sample_pair(patch_rotated)
            samples.append(vin)
            labels.append(vout)

            patch_rotated = np.rot90(patch_hor_flip, k=idx)
            vin, vout = gen_sample_pair(patch_rotated)
            samples.append(vin)
            labels.append(vout)

    samples = np.asarray(samples, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)

    Path(FLAGS.save_dir).mkdir(parents=True, exist_ok=True)
    out_file = f'{FLAGS.save_dir}/{m}.hdf5'
    with h5py.File(out_file, 'w') as f:
        print(f'Written file: {m}')
        for i in range(num_patches):
            f.create_dataset(f'images_{i}', data=samples[i],
                             compression='gzip')
            f.create_dataset(f'labels_{i}', data=labels[i],
                             compression='gzip')


def lol(lst, sz):
    return [lst[i: i + sz] for i in range(0, len(lst), sz)]


def main(_):
    num_patches = FLAGS.num_patches
    patch_dim = FLAGS.patch_dim

    root_dir = FLAGS.root_dir
    root_dir = Path(root_dir)

    train_dir = root_dir.joinpath('train')
    val_dir = root_dir.joinpath('val')

    train_images = list(paths.list_images(train_dir))
    val_images = list(paths.list_images(val_dir))

    print(f'Len of train: {len(train_images)}')
    print(f'Len of val: {len(val_images)}')

    sample_shape = (3, patch_dim, patch_dim, num_patches)

    num_chunks = 10
    num_batches = 101
    chunks = lol(np.arange(num_batches), num_chunks)
    train_chunks = lol(np.arange(len(train_images)), len(train_images)
                       // num_chunks)

    manager = Manager()
    samples = manager.list([[]] * num_batches)
    labels = manager.list([[]] * num_batches)

    for i, c in enumerate(chunks):
        train_chunk = train_chunks[min(i, len(train_chunks) - 1)]
        jobs = []
        train_imgs_chunk = [train_images[i] for i in train_chunk]
        for m in c:
            p = multiprocessing.Process(target=generate,
                                        args=[(train_imgs_chunk,
                                               val_images,
                                               samples, labels,
                                               patch_dim, num_patches,
                                               m)])
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()


if __name__ == "__main__":
    app.run(main)
