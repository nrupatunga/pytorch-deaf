"""
File: test_new.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com Github: https://github.com/nrupatunga
Description: Test script
"""
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.fftpack import fft2, ifft2

from litdeaf import deafLitModel
from psf2otf import psf2otf

SCALE_INPUT = 1.0


def get_h_input(S):
    h = np.diff(S, axis=1)
    last_col = S[:, 0, :] - S[:, -1, :]
    last_col = last_col[:, np.newaxis, :]

    h = np.hstack([h, last_col])
    return h / SCALE_INPUT


def get_v_input(S):
    v = np.diff(S, axis=0)
    last_row = S[0, ...] - S[-1, ...]
    last_row = last_row[np.newaxis, ...]

    v = np.vstack([v, last_row])
    return v / SCALE_INPUT


def get_outputs(ckpt_path, img_path, gpu=True):
    S_out = []
    S = cv2.imread(img_path) / 255.
    S_out.append(S)
    for i in range(2):
        if i == 0:
            h_input = get_h_input(S)
            data_in = np.transpose(h_input, [1, 0, 2])
        else:
            v_input = get_v_input(S)
            data_in = v_input

        data = np.transpose(data_in, [2, 0, 1])
        data = torch.tensor(data).unsqueeze(0)
        data = data.float()

        # Model loading from checkpoint
        if gpu:
            model = deafLitModel.load_from_checkpoint(ckpt_path).cuda()
        else:
            model = deafLitModel.load_from_checkpoint(ckpt_path,
                                                      map_location=torch.device('cpu'))
        model.eval()
        model.freeze()

        if gpu:
            output = model._model(data.cuda()).squeeze().cpu().numpy()
        else:
            output = model._model(data).squeeze().numpy()

        output = np.transpose(output, [1, 2, 0])
        out = np.zeros_like(data_in)
        out[4:4 + output.shape[0], 4:4 + output.shape[1], :] = output
        if i == 1:
            final_out = np.hstack((S_out[0], out))
            S_out.append(out)
        else:
            data_in = np.transpose(data_in, [1, 0, 2])
            out = np.transpose(out, [1, 0, 2])
            final_out = np.hstack((S_out[0], out))
            S_out.append(out)

        if False:
            plt.imshow(final_out, cmap=plt.cm.Blues)
            plt.draw()
            plt.waitforbuttonpress(0)
            plt.close()

    return S_out


def main(args):
    ckpt_path = args.ckpt_path
    img_path = args.img_path

    [S, h, v] = get_outputs(ckpt_path, img_path, args.gpu)

    h = h * SCALE_INPUT
    v = v * SCALE_INPUT

    beta = 8.388608e+1 / 2.
    S_in = S

    psf = np.asarray([[-1, 1]])
    out_size = (S.shape[0], S.shape[1])
    otfx = psf2otf(psf, out_size)
    psf = np.asarray([[-1], [1]])
    otfy = psf2otf(psf, out_size)

    Normin1 = fft2(np.squeeze(S), axes=(0, 1))
    Denormin2 = np.square(abs(otfx)) + np.square(abs(otfy))
    Denormin2 = Denormin2[..., np.newaxis]
    Denormin2 = np.repeat(Denormin2, 3, axis=2)
    Denormin = 1 + beta * Denormin2

    h_diff = -np.diff(h, axis=1)
    first_col = h[:, -1, :] - h[:, 0, :]
    first_col = first_col[:, np.newaxis, :]
    h_diff = np.hstack([first_col, h_diff])

    v_diff = -np.diff(v, axis=0)
    first_row = v[-1, ...] - v[0, ...]
    first_row = first_row[np.newaxis, ...]
    v_diff = np.vstack([first_row, v_diff])

    Normin2 = h_diff + v_diff
    # Normin2 = beta * np.fft.fft2(Normin2, axes=(0, 1))
    Normin2 = beta * fft2(Normin2, axes=(0, 1))

    Normin1 = fft2(np.squeeze(S), axes=(0, 1))
    FS = np.divide(np.squeeze(Normin1) + np.squeeze(Normin2),
                   Denormin)
    # S = np.real(np.fft.ifft2(FS, axes=(0, 1)))
    S = np.real(ifft2(FS, axes=(0, 1)))

    S = np.squeeze(S)
    S = np.clip(S, 0, 1)
    S = S * 255
    S = S.astype(np.uint8)
    cv2.imwrite('output.png', S)
    S = cv2.cvtColor(S, cv2.COLOR_BGR2RGB)
    S_in = S_in * 255
    S_in = S_in.astype(np.uint8)
    S_in = cv2.cvtColor(S_in, cv2.COLOR_BGR2RGB)
    plt.imshow(np.hstack((S_in, S)))
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()


def get_args():
    """get specific args for testings"""
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        '--ckpt_path',
        type=str,
        help='ckpt path')

    parser.add_argument('--gpu', action='store_false', help='gpu/cpu')
    parser.add_argument(
        '--img_path',
        type=str,
        help='image path')

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
