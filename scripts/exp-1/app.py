"""
File: app.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: Demo app for deaf project
"""

import gradio as gr
import numpy as np
import torch
from scipy.fftpack import fft2, ifft2

from litdeaf import deafLitModel
from src.utility.psf2otf import psf2otf

# load model
gpu = True
ckpt_path = './ckpt/deaf_deaf__ckpt_epoch_8.ckpt'
# Model loading from checkpoint
if gpu:
    model = deafLitModel.load_from_checkpoint(ckpt_path).cuda()
else:
    model = deafLitModel.load_from_checkpoint(ckpt_path,
                                              map_location=torch.device('cpu'))

model.eval()
model.freeze()

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


def get_output(S):
    S_out = []
    S = S / 255.
    S_out.append(S)
    for i in range(2):
        if i == 0:
            h_input = get_h_input(S)
            data_in = np.transpose(h_input, [1, 0, 2])
        else:
            v_input = get_v_input(S)
            data_in = v_input

        data = np.transpose(data_in, [2, 0, 1])
        data = data - 8.8276e-7
        data = data / 0.1637
        data = torch.tensor(data).unsqueeze(0)
        data = data.float()

        if gpu:
            output = model._model(data.cuda()).squeeze().cpu().numpy()
        else:
            output = model._model(data).squeeze().numpy()

        output = np.transpose(output, [1, 2, 0])
        out = np.zeros_like(data_in)
        out[4:4 + output.shape[0], 4:4 + output.shape[1], :] = output
        if i == 1:
            S_out.append(out)
        else:
            data_in = np.transpose(data_in, [1, 0, 2])
            out = np.transpose(out, [1, 0, 2])
            S_out.append(out)

    return S_out


def EdgeAwareFilterOutput(img):

    [S, h, v] = get_output(img)
    h = h * SCALE_INPUT
    v = v * SCALE_INPUT

    beta = 8.388608e+1 / 2.

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
    # cv2.imwrite('output.png', S)
    return S


inputs = gr.inputs.Image()
outputs = gr.outputs.Image()
sample_images = [
    ['./images/pflower.jpg'],
    ['./images/flower.png'],
    ['./images/basketball.png'],
    ['./images/216053.jpg'],
    ['./images/rock2.png'],
]

gr.Interface(fn=EdgeAwareFilterOutput,
             inputs=inputs,
             outputs=outputs,
             description='Demo - Deep Edge Aware Filter (L0-Smoothing) using CNN',
             examples=sample_images).launch()
