import numpy as np
import dill
import torch
from functools import partial
from utils import (linear_scale_forward, UniformPrior, simulator_hnn)
from dask_jobqueue import SLURMCluster
from hnn_core import jones_2009_model
from distributed import Client

device = 'cpu'

# Set up cluster and reserve resources
cluster = SLURMCluster(
    cores=32, processes=32, queue='compute', memory="256GB", walltime="3:00:00",
    job_extra=['-A csd403', '--nodes=1'], log_directory=Path.cwd() /'slurm_out')

client = Client(cluster)
print(client.dashboard_link)

num_cores = 256
step_size = num_cores
client.cluster.scale(num_cores)

net = jones_2009_model()
net.clear_connectivity()

# Number of simulations to run when sampling from prior
num_sims = 110_000
   
save_path = '../data/hnn_rc/sbi_sims'
temp_path = '../data/hnn_rc/temp'
    
prior_dict = {'prox_weight': {'bounds': (-4, -3), 'scale_func': log_scale_forward},
              'dist_weight': {'bounds': (-4, -3), 'scale_func': log_scale_forward}, 
              'latency': {'bounds': (-75, 75), 'scale_func': linear_scale_forward}}

# Create uniform prior and sample
prior = UniformPrior(parameters=list(prior_dict.keys()))
theta_samples = prior.sample((num_sims,))

with open(f'{save_path}/prior_dict.pkl', 'wb') as f:
    dill.dump(prior_dict, f)

sim_metadata = {'tstop': 350, 'dt': 0.5}
with open(f'{save_path}/sim_metadata.pkl', 'wb') as f:
    dill.dump(sim_metadata, f)
    
    
# Create batch simulation function
def batch(seq, theta_samples, save_path):
    # create simulator object, rescale function transforms (0,1) to range specified in prior_dict
    simulator = partial(simulator_hnn, prior_dict=prior_dict, param_function=param_function,
                        rescale_function=scale_func_list, network_model=net)

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
for i in range(0, num_sims, step_size):
    print(i)
    if i + step_size < theta_samples.shape[0]:
        batch(list(range(i, i + step_size)), theta_samples[i:i + step_size, :], temp_path_save)
    else:
        print(i, theta_samples.shape[0])
        batch(list(range(i, theta_samples.shape[0])), theta_samples[i:, :], temp_path_save)

# Load simulations into single array, save output, and remove small small files
x_orig, theta_orig = load_prerun_simulations(str(temp_path_save) + '/')
x_name = f'{save_path}/dpl_sim_uniform.npy'
theta_name = f'{save_path}/theta_sim_uniform.npy'
np.save(x_name, x_orig)
np.save(theta_name, theta_orig)

files = glob.glob(str(temp_path_save) + '/*')
for f in files:
    os.remove(f)


