import sys
sys.path.append('../')
import os
import numpy as np
import dill
import torch
from utils import run_rc_sim
from itertools import product

device = 'cpu'

save_path = '../../data/rc_circuit'
save_suffix = 'grid'
    
with open(f'{save_path}/sbi_sims/prior_dict.pkl', 'rb') as output_file:
    prior_dict = dill.load(output_file)
with open(f'{save_path}/sbi_sims/sim_metadata.pkl', 'rb') as output_file:
    sim_metadata = dill.load(output_file)
    
tstop = sim_metadata['tstop']
    
n_params = len(prior_dict)
    
# Evenly spaced grid on (0,1) for theta samples (mapped to bounds defined in prior_dict during simulation)
n_points = 10
sample_points = [np.linspace(0.05, 0.95, n_points).tolist() for _ in range(n_params)]
theta_samples = list(product(sample_points[0], sample_points[1], sample_points[2]))
theta_samples = torch.tensor(theta_samples)

run_rc_sim(prior_dict, theta_samples, tstop, save_path, save_suffix)

#os.system('scancel -u ntolley')

