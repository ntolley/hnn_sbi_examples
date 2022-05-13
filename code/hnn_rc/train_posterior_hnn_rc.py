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

data_path = '../../data/hnn_rc'

#Number of samples to set to zero
zero_samples = 100
noise_amp = 1e-5
train_posterior(data_path, ntrain_sims, noise_amp, zero_samples)

   
#os.system('scancel -u ntolley')
