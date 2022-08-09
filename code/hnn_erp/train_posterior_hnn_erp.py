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

ntrain_sims = 100_000

data_path = '../../data/hnn_erp'

# Window specifying portion of time series for inference
window_samples = [0, -1]
x_noise_amp = 1e-5
theta_noise_amp = 0.0
train_posterior(data_path, ntrain_sims, x_noise_amp, theta_noise_amp, window_samples)

   
#os.system('scancel -u ntolley')
