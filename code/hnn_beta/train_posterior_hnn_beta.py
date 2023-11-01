import sys
sys.path.append('../')
import torch
import os
import numpy as np
from functools import partial

from sbi import inference as sbi_inference
from utils import train_posterior
import pickle
import dill

device = 'cpu'

ntrain_sims = 10_000

data_path = '../../data/hnn_beta'

#Number of samples to set to zero
window_samples = (316, 516)
x_noise_amp = 1e-5
theta_noise_amp = 0.0

# Signals are really tiny, need to scale and look at high frequency bands
# Otherwise log(bandpower) = log(0.0) = -np.inf
scale_factor=60_000
freq_band_list=[(13,30), (30,50), (50,80)]

train_posterior(data_path, ntrain_sims, x_noise_amp, theta_noise_amp, window_samples,
                scale_factor=scale_factor, freq_band_list=freq_band_list)

