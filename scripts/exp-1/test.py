"""
File: test.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: testing
"""
import cv2
import numpy as np
import torch

from litdeaf import deafLitModel

dbg = False


img_path = \
    '/home/nthere/2020/vcnn_double-bladed/applications/deep_edge_aware_filters/images/1.png'

S = cv2.imread(img_path) / 255.


def get_h_input(S):
    h = np.diff(S, axis=1)
    last_col = S[:, 0, :] - S[:, -1, :]
    last_col = last_col[:, np.newaxis, :]

    h = np.hstack([h, last_col])
    return h * 2


def get_v_input(S):
    v = np.diff(S, axis=0)
    last_row = S[0, ...] - S[-1, ...]
    last_row = last_row[np.newaxis, ...]

    v = np.vstack([v, last_row])
    return v * 2


v_input = get_v_input(S)
h_input = get_h_input(S)
if dbg:
    v_norm = cv2.normalize(v_input, None, alpha=0, beta=255,
                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    h_norm = cv2.normalize(h_input, None, alpha=0, beta=255,
                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    cv2.imwrite('h.jpg', h_norm)
    cv2.imwrite('v.jpg', v_norm)


ckpt_path = './deaf/deaf_epoch=19.ckpt'
data = h_input[0:270, 0:270]
cv2.imwrite('input.jpg', data * 255)
data = np.transpose(data, [2, 0, 1])
# data = data / 255.
data = torch.tensor(data).unsqueeze(0)
data = data.float()
model = deafLitModel.load_from_checkpoint(ckpt_path).cuda()
model.eval()
model.freeze()
output = model._model(data.cuda()).squeeze().cpu().numpy()
output = np.transpose(output, [1, 2, 0])
output = output * 255
cv2.imwrite('output.jpg', output)
