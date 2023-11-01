import sys
sys.path.append('../')
import os
import dill
from utils import (linear_scale_forward, log_scale_forward, start_cluster,
                   run_hnn_sim, hnn_beta_param_function, UniformPrior)
from hnn_core import jones_2009_model

nsbi_sims = 100_000
tstop = 500
dt = 0.5

net = jones_2009_model()
   
save_path = '../../data/hnn_beta/'
save_suffix = 'sbi'
    
# prior_dict = {'dist_var': {'bounds': (0, 20), 'rescale_function': linear_scale_forward},
#               'prox_var': {'bounds': (20, 40), 'rescale_function': linear_scale_forward},
#               'dist_exc': {'bounds': (-6, -4), 'rescale_function': log_scale_forward},
#               'prox_exc': {'bounds': (-6, -4), 'rescale_function': log_scale_forward},}

prior_dict = {'dist_var': {'bounds': (0,20), 'rescale_function': linear_scale_forward},
              'prox_var': {'bounds': (0,40), 'rescale_function': linear_scale_forward},}

with open(f'{save_path}/sbi_sims/prior_dict.pkl', 'wb') as f:
    dill.dump(prior_dict, f)

sim_metadata = {'nsbi_sims': nsbi_sims, 'tstop': tstop, 'dt': dt}
with open(f'{save_path}/sbi_sims/sim_metadata.pkl', 'wb') as f:
    dill.dump(sim_metadata, f)

prior = UniformPrior(parameters=list(prior_dict.keys()))
theta_samples = prior.sample((nsbi_sims,))

start_cluster() # reserve resources for HNN simulations

run_hnn_sim(net=net, param_function=hnn_beta_param_function, prior_dict=prior_dict,
            theta_samples=theta_samples, tstop=tstop, save_path=save_path, save_suffix=save_suffix)

