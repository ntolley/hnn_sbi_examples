import sys
import os
import numpy as np
import dill
import torch
from functools import partial
from utils import (linear_scale_forward, log_scale_forward, UniformPrior,
                   simulator_hnn, hnn_beta_param_function, load_prerun_simulations)
from dask_jobqueue import SLURMCluster
import dask
from hnn_core import jones_2009_model
from distributed import Client
import glob

device = 'cpu'

# Set up cluster and reserve resources
cluster = SLURMCluster(
    cores=32, processes=32, queue='compute', memory="256GB", walltime="3:00:00",
    job_extra=['-A csd403', '--nodes=1'], log_directory=os.getcwd() + '/slurm_out')

client = Client(cluster)
print(client.dashboard_link)

num_cores = 256
step_size = num_cores
client.cluster.scale(num_cores)

net = jones_2009_model()

# Number of simulations to run when sampling from prior
num_sims = 110_000
   
save_path = '../../data/hnn_beta/sbi_sims'
temp_path = '../../data/hnn_beta/temp'
    
prior_dict = {'dist_var': {'bounds': (0, 40), 'rescale_function': linear_scale_forward},
              'prox_var': {'bounds': (0, 40), 'rescale_function': linear_scale_forward},
              'dist_exc': {'bounds': (-7, -4), 'rescale_function': log_scale_forward},
              'prox_exc': {'bounds': (-7, -4), 'rescale_function': log_scale_forward},}

# Create uniform prior and sample
prior = UniformPrior(parameters=list(prior_dict.keys()))
theta_samples = prior.sample((num_sims,))

with open(f'{save_path}/prior_dict.pkl', 'wb') as f:
    dill.dump(prior_dict, f)

sim_metadata = {'tstop': 500, 'dt': 0.5}
with open(f'{save_path}/sim_metadata.pkl', 'wb') as f:
    dill.dump(sim_metadata, f)
    
    
# Create batch simulation function
def batch(seq, theta_samples, save_path):
    # create simulator object, rescale function transforms (0,1) to range specified in prior_dict
    simulator = partial(simulator_hnn, prior_dict=prior_dict, param_function=hnn_beta_param_function,
                        network_model=net)

    # Create lazy list of tasks
    res_list= []
    for sim_idx in range(len(seq)):
        res = dask.delayed(simulator)(theta_samples[sim_idx,:])
        res_list.append(res)

    # Run tasks
    final_res = dask.compute(*res_list)
    x_list = np.stack([final_res[idx][0] for idx in range(len(seq))])

    x_name = f'{temp_path}/x_sbi{seq[0]}-{seq[-1]}.npy'
    theta_name = f'{temp_path}/theta_sbi{seq[0]}-{seq[-1]}.npy'

    np.save(x_name, x_list)
    np.save(theta_name, theta_samples.detach().cpu().numpy())

# Generate simulations
seq_list = list()
for i in range(0, num_sims, step_size):
    print(i)
    seq = list(range(i, i + step_size))
    if i + step_size < theta_samples.shape[0]:
        batch(seq, theta_samples[i:i + step_size, :], temp_path)
    else:
        print(i, theta_samples.shape[0])
        seq = list(range(i, theta_samples.shape[0]))
        batch(seq, theta_samples[i:, :], temp_path)
    seq_list.append(seq)

# Load simulations into single array, save output, and remove small small files
x_files = [f'{temp_path}/x_sbi{seq[0]}-{seq[-1]}.npy' for seq in seq_list]
theta_files = [f'{temp_path}/theta_sbi{seq[0]}-{seq[-1]}.npy' for seq in seq_list]

x_orig, theta_orig = load_prerun_simulations(x_files, theta_files)
x_name = f'{save_path}/x_sbi.npy'
theta_name = f'{save_path}/theta_sbi.npy'
np.save(x_name, x_orig)
np.save(theta_name, theta_orig)

files = glob.glob(str(temp_path) + '/*')
for f in files:
    os.remove(f)


os.system('scancel -u ntolley')