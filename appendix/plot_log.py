import json
import sys

import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    with open(sys.argv[1], 'r', encoding='utf8') as f:
        log = np.asarray(json.load(f))
    print(np.min(log, axis=0))
    log = log[:(log.shape[0] // 4) * 4]
    split_trn = np.array_split(log[:, 0], log.shape[0] // 4)
    split_val = np.array_split(log[:, 1], log.shape[0] // 4)

    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 12

    mean_val = np.mean(split_val, axis=1)
    min_val = np.min(split_val, axis=1)
    std_val = np.std(split_val, axis=1)
    x_val = np.arange(len(mean_val))

    fig = plt.figure()

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.fill_between(
        x_val, mean_val - std_val, mean_val + std_val, alpha=0.5, color='r')
    ax1.plot(x_val, mean_val, label='validation mean', c='r')
    ax1.plot(x_val, min_val, label='validation min', c='k', ls='--')

    ax1.grid(which='both', color='gray', linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(edgecolor='white')

    mean_trn = np.mean(split_trn, axis=1)
    std_trn = np.std(split_trn, axis=1)
    x_trn = np.arange(len(mean_trn))

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.fill_between(
        x_trn, mean_trn - std_trn, mean_trn + std_trn, alpha=0.5, color='b')
    ax2.plot(x_trn, mean_trn, label='training mean', c='b')

    ax2.grid(which='both', color='gray', linestyle='--')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(edgecolor='white')

    fig.tight_layout()
    plt.show()
