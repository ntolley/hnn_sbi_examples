import sys
sys.path.append('../')
import os
from utils import (linear_scale_forward, log_scale_forward,
                   run_hnn_sim, hnn_rc_param_function, UniformPrior)
from hnn_core import jones_2009_model

nsbi_sims = 110_0

net = jones_2009_model()
net.clear_connectivity()

   
save_path = '../../data/hnn_rc/'
save_suffix = 'sbi'
    
prior_dict = {'prox_weight': {'bounds': (-4, -3), 'rescale_function': log_scale_forward},
              'dist_weight': {'bounds': (-4, -3), 'rescale_function': log_scale_forward}, 
              'latency': {'bounds': (-75, 75), 'rescale_function': linear_scale_forward}}

prior = UniformPrior(parameters=list(prior_dict.keys()))
theta_samples = prior.sample((nsbi_sims,))

tstop = 350

run_hnn_sim(net=net, param_function=hnn_rc_param_function, prior_dict=prior_dict,
            theta_samples=theta_samples, tstop=tstop, save_path=save_path, save_suffix=save_suffix)

#os.system('scancel -u ntolley')