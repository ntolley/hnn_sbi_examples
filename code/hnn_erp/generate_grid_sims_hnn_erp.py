import sys
import os
import numpy as np
import dill
import torch
from functools import partial
from utils import (log_scale_forward, UniformPrior,
                   simulator_hnn, hnn_erp_param_function, load_prerun_simulations)
from dask_jobqueue import SLURMCluster
import dask
from hnn_core import jones_2009_model
from distributed import Client
import glob
from itertools import product

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
net.clear_connectivity()

save_path = '../../data/hnn_erp/sbi_sims'
temp_path = '../../data/hnn_erp/temp'
    
with open(f'{save_path}/prior_dict.pkl', 'rb') as output_file:
    prior_dict = dill.load(output_file)
with open(f'{save_path}/sim_metadata.pkl', 'rb') as output_file:
    sim_metadata = dill.load(output_file)
    
n_params = len(prior_dict)
    
# Evenly spaced grid on (0,1) for theta samples (mapped to bounds defined in prior_dict during simulation)
n_points = 10
sample_points = [np.linspace(1e-10,1, n_points).tolist() for _ in range(n_params)]
theta_samples = list(product(sample_points[0], sample_points[1], sample_points[2], sample_points[3]))
theta_samples = torch.tensor(theta_samples)
num_sims = theta_samples.shape[0]

# Create batch simulation function
def batch(seq, theta_samples, save_path):
    # create simulator object, rescale function transforms (0,1) to range specified in prior_dict
    simulator = partial(simulator_hnn, prior_dict=prior_dict, param_function=hnn_erp_param_function,
                        network_model=net)

    # Create lazy list of tasks
    res_list= []
    for sim_idx in range(len(seq)):
        res = dask.delayed(simulator)(theta_samples[sim_idx,:])
        res_list.append(res)

    # Run tasks
    final_res = dask.compute(*res_list)
    x_list = np.stack([final_res[idx][0] for idx in range(len(seq))])

    x_name = f'{temp_path}/x_grid{seq[0]}-{seq[-1]}.npy'
    theta_name = f'{temp_path}/theta_grid{seq[0]}-{seq[-1]}.npy'

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
x_files = [f'{temp_path}/x_grid{seq[0]}-{seq[-1]}.npy' for seq in seq_list]
theta_files = [f'{temp_path}/theta_grid{seq[0]}-{seq[-1]}.npy' for seq in seq_list]

x_orig, theta_orig = load_prerun_simulations(x_files, theta_files)
x_name = f'{save_path}/x_grid.npy'
theta_name = f'{save_path}/theta_grid.npy'
np.save(x_name, x_orig)
np.save(theta_name, theta_orig)

files = glob.glob(str(temp_path) + '/*')
for f in files:
    os.remove(f)


#os.system('scancel -u ntolley')

