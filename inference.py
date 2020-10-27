import argparse
import os

import librosa
import numpy as np
import soundfile as sf
import torch

from lib import nets
from lib import spec_utils
from lib import utils


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--pretrained_model', '-P', type=str, default='models/baseline.pth')
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--hop_length', '-l', type=int, default=1024)
    p.add_argument('--window_size', '-w', type=int, default=512)
    p.add_argument('--output_image', '-I', action='store_true')
    p.add_argument('--postprocess', '-p', action='store_true')
    p.add_argument('--tta', '-t', action='store_true')
    args = p.parse_args()

    print('loading model...', end=' ')
    device = torch.device('cpu')
    model = nets.CascadedASPPNet(args.n_fft)
    model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)
    print('done')

    print('loading wave source...', end=' ')
    X, sr = librosa.load(
        args.input, args.sr, False, dtype=np.float32, res_type='kaiser_fast')
    basename = os.path.splitext(os.path.basename(args.input))[0]
    print('done')

    if X.ndim == 1:
        X = np.asarray([X, X])

    print('stft of wave source...', end=' ')
    X = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)
    print('done')

    vr = utils.VocalRemover(model, device, args.window_size)

    if args.tta:
        pred_mag, pred_phase, X_mag, X_phase = vr.inference_tta(X)
    else:
        pred_mag, pred_phase, X_mag, X_phase = vr.inference(X)

    if args.postprocess:
        print('post processing...', end=' ')
        pred_inv = np.clip(X_mag - pred_mag, 0, np.inf)
        pred_mag = spec_utils.mask_silence(pred_mag, pred_inv)
        print('done')

    print('inverse stft of instruments...', end=' ')
    y_spec = pred_mag * np.exp(1.j * pred_phase)
    wave = spec_utils.spectrogram_to_wave(y_spec, hop_length=args.hop_length)
    print('done')
    sf.write('{}_Instruments.wav'.format(basename), wave.T, sr)

    print('inverse stft of vocals...', end=' ')
    v_spec = X - y_spec
    wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=args.hop_length)
    print('done')
    sf.write('{}_Vocals.wav'.format(basename), wave.T, sr)

    if args.output_image:
        image = spec_utils.spectrogram_to_image(y_spec)
        utils.imwrite('{}_Instruments.jpg'.format(basename), image)

        image = spec_utils.spectrogram_to_image(np.angle(X) - pred_phase, mode='phase')
        utils.imwrite('{}_Phase.jpg'.format(basename), image)

        image = spec_utils.spectrogram_to_image(v_spec)
        utils.imwrite('{}_Vocals.jpg'.format(basename).format(basename), image)


if __name__ == '__main__':
    main()
