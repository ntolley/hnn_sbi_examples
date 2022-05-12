import sys
sys.path.append('../')
import os
#sys.path.append(os.get_cwd())
from utils import (linear_scale_forward, log_scale_forward, run_hnn_sim,
                   simulator_hnn, hnn_rc_param_function, load_prerun_simulations)
from hnn_core import jones_2009_model

net = jones_2009_model()
net.clear_connectivity()

# Number of simulations to run when sampling from prior
   
save_path = '../../data/hnn_rc/'
    
prior_dict = {'prox_weight': {'bounds': (-4, -3), 'rescale_function': log_scale_forward},
              'dist_weight': {'bounds': (-4, -3), 'rescale_function': log_scale_forward}, 
              'latency': {'bounds': (-75, 75), 'rescale_function': linear_scale_forward}}
tstop = 350

run_hnn_sim(net=net, param_function=hnn_rc_param_function, prior_dict=prior_dict, tstop=tstop, save_path=save_path)

os.system('scancel -u ntolley')