import sys
sys.path.append('../')
import numpy as np
import dill
import torch
from utils import (linear_scale_forward, UniformPrior, run_rc_sim)

# Number of simulations to run when sampling from prior
num_sims = 110_000
   
save_path = '../data/rc_circuit'
    
prior_dict = {'amp1': {'bounds': (0, 1), 'scale_func': linear_scale_forward},
              'amp2': {'bounds': (-1, 0), 'scale_func': linear_scale_forward}, 
              'latency': {'bounds': (-20, 20), 'scale_func': linear_scale_forward}}

# Create uniform prior and sample
prior = UniformPrior(parameters=list(prior_dict.keys()))
theta_samples = prior.sample((num_sims,))

with open(f'{save_path}/prior_dict.pkl', 'wb') as f:
    dill.dump(prior_dict, f)

sim_metadata = {'tstop': 80, 'dt': 0.5}
with open(f'{save_path}sbi_sims//sim_metadata.pkl', 'wb') as f:
    dill.dump(sim_metadata, f)

v_list = list()
for sim_idx in range(num_sims):
    if sim_idx % 1000 == 0:
        print(sim_idx, end=' ')
    thetai = theta_samples[sim_idx, :]
    theta_dict = {param_name: param_dict['scale_func'](thetai[idx].numpy(), param_dict['bounds']) for 
                  idx, (param_name, param_dict) in enumerate(prior_dict.items())}

    v_out = run_rc_sim(theta_dict, tstop=sim_metadata['tstop'], dt=sim_metadata['dt'])
    v_list.append(v_out)

# Save simulation output
x_sims = np.hstack(v_list).T
theta_sims = theta_samples.numpy()

x_name = f'{save_path}/sbi_sims/x_sbi.npy'
theta_name = f'{save_path}/sbi_sims/theta_sbi.npy'
np.save(x_name, x_sims)
np.save(theta_name, theta_sims)

