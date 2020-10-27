import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from lib import dataset


class VocalRemover(object):

    def __init__(self, model, device, window_size):
        self.model = model
        self.offset = model.offset
        self.device = device
        self.window_size = window_size

    def _execute(self, X_mag, X_phase, roi_size, n_window):
        self.model.eval()
        with torch.no_grad():
            preds_mag = []
            preds_phase = []
            for i in tqdm(range(n_window)):
                start = i * roi_size
                X_mag_window = X_mag[None, :, :, start:start + self.window_size]
                X_phase_window = X_phase[None, :, :, start:start + self.window_size]

                X_mag_window = torch.from_numpy(X_mag_window).to(self.device)
                X_phase_window = torch.from_numpy(X_phase_window).to(self.device)

                pred_mag, pred_phase = self.model.predict(X_mag_window, X_phase_window)

                pred_mag = pred_mag.detach().cpu().numpy()
                pred_phase = pred_phase.detach().cpu().numpy()
                preds_mag.append(pred_mag[0])
                preds_phase.append(pred_phase[0])

            pred_mag = np.concatenate(preds_mag, axis=2)
            pred_phase = np.concatenate(preds_phase, axis=2)

        return pred_mag, pred_phase

    def preprocess(self, X_spec):
        X_mag = np.abs(X_spec)
        X_phase = np.angle(X_spec)

        return X_mag, X_phase

    def _inference(self, X_mag_pre=None, X_phase_pre=None, pad=(0, 0), roi_size=0, n_window=0):
        X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), pad), mode='constant')
        X_phase_pad = np.pad(X_phase_pre, ((0, 0), (0, 0), pad), mode='constant')

        pred_mag, pred_phase = self._execute(X_mag_pad, X_phase_pad, roi_size, n_window)

        return pred_mag, pred_phase

    def inference(self, X_spec):
        X_mag, X_phase = self.preprocess(X_spec)

        coef = X_mag.max()
        X_mag_pre = X_mag / coef
        X_phase_pre = X_phase / np.pi

        n_frame = X_mag_pre.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.window_size, self.offset)
        n_window = int(np.ceil(n_frame / roi_size))

        pred_mag, pred_phase = self._inference(
            X_mag_pre, X_phase_pre, (pad_l, pad_r), roi_size, n_window)

        pred_mag = pred_mag[:, :, :n_frame] * coef
        pred_phase = pred_phase[:, :, :n_frame] * np.pi

        return pred_mag, pred_phase, X_mag, X_phase

    def inference_tta(self, X_spec):
        X_mag, X_phase = self.preprocess(X_spec)

        coef = X_mag.max()
        X_mag_pre = X_mag / coef
        X_phase_pre = X_phase / np.pi

        n_frame = X_mag_pre.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.window_size, self.offset)
        n_window = int(np.ceil(n_frame / roi_size))

        pred_mag, pred_phase = self._inference(
            X_mag_pre, X_phase_pre, (pad_l, pad_r), roi_size, n_window)

        pad_l += roi_size // 2
        pad_r += roi_size // 2
        n_window += 1

        pred_mag_tta, pred_phase_tta = self._inference(
            X_mag_pre, X_phase_pre, (pad_l, pad_r), roi_size, n_window)

        pred_mag_tta = pred_mag_tta[:, :, roi_size // 2:]
        pred_phase_tta = pred_phase_tta[:, :, roi_size // 2:]

        pred_mag = (pred_mag[:, :, :n_frame] + pred_mag_tta[:, :, :n_frame]) * 0.5 * coef
        pred_phase = (pred_phase[:, :, :n_frame] + pred_phase_tta[:, :, :n_frame]) * 0.5 * np.pi

        return pred_mag, pred_phase, X_mag, X_phase


def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False
