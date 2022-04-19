import sys
sys.path.append('../')
import numpy as np
import dill
import torch
from itertools import product
from utils import (linear_scale_forward, UniformPrior, run_rc_sim)

save_path = '../data/rc_circuit/sbi_sims'
    
with open(f'{save_path}/prior_dict.pkl', 'rb') as output_file:
    prior_dict = dill.load(output_file)
with open(f'{save_path}/sim_metadata.pkl', 'rb') as output_file:
    sim_metadata = dill.load(output_file)
    
n_params = len(prior_dict)
    
# Evenly spaced grid on (0,1) for theta samples (mapped to bounds defined in prior_dict during simulation)
n_points = 10
sample_points = [np.linspace(1e-10,1, n_points).tolist() for _ in range(n_params)]
theta_samples = list(product(sample_points[0], sample_points[1], sample_points[2]))
theta_samples = np.array(theta_samples)
num_sims = theta_samples.shape[0]

v_list = list()
for sim_idx in range(num_sims):
    if sim_idx % 1000 == 0:
        print(sim_idx, end=' ')
    thetai = theta_samples[sim_idx, :]
    theta_dict = {param_name: param_dict['scale_func'](thetai[idx], param_dict['bounds']) for 
                  idx, (param_name, param_dict) in enumerate(prior_dict.items())}

    v_out = run_rc_sim(theta_dict, tstop=sim_metadata['tstop'], dt=sim_metadata['dt'])
    v_list.append(v_out)

# Save simulation output
x_sims = np.hstack(v_list).T

x_name = f'{save_path}/x_grid.npy'
theta_name = f'{save_path}/theta_grid.npy'
np.save(x_name, x_sims)
np.save(theta_name, theta_samples)

