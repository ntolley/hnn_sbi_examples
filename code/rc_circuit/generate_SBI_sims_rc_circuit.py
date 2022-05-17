import sys
sys.path.append('../')
import numpy as np
import dill
import torch
from utils import (linear_scale_forward, UniformPrior, run_rc_sim)

# Number of simulations to run when sampling from prior
nsbi_sims = 110_000
tstop = 80
dt = 0.5
   
save_path = '../../data/rc_circuit'
save_suffix = 'sbi'

prior_dict = {'amp1': {'bounds': (0, 1), 'scale_func': linear_scale_forward},
              'amp2': {'bounds': (-1, 0), 'scale_func': linear_scale_forward}, 
              'latency': {'bounds': (-20, 20), 'scale_func': linear_scale_forward}}

# Create uniform prior and sample
prior = UniformPrior(parameters=list(prior_dict.keys()))
theta_samples = prior.sample((nsbi_sims,))

with open(f'{save_path}/sbi_sims/prior_dict.pkl', 'wb') as f:
    dill.dump(prior_dict, f)

sim_metadata = {'nsbi_sims': nsbi_sims, 'tstop': tstop, 'dt': dt}
with open(f'{save_path}/sbi_sims//sim_metadata.pkl', 'wb') as f:
    dill.dump(sim_metadata, f)

run_rc_sim(prior_dict, theta_samples, tstop, save_path, save_suffix)

#os.system('scancel -u ntolley')

