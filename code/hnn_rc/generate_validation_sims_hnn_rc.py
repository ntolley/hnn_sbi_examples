import sys
import os
import numpy as np
import dill
import torch
from functools import partial
from utils import (linear_scale_forward, log_scale_forward, UniformPrior,
                   simulator_hnn, hnn_rc_param_function, load_prerun_simulations,
                   get_dataset_psd, get_dataset_peaks, load_posterior)
from sklearn.decomposition import PCA
from dask_jobqueue import SLURMCluster
import dask
from hnn_core import jones_2009_model
from distributed import Client
import glob
from itertools import product

device = 'cpu'

# Set up cluster and reserve resources
cluster = SLURMCluster(
    cores=32, processes=32, queue='compute', memory="256GB", walltime="24:00:00",
    job_extra=['-A csd403', '--nodes=1'], log_directory=os.getcwd() + '/slurm_out')

client = Client(cluster)
print(client.dashboard_link)

num_cores = 128
step_size = num_cores
client.cluster.scale(num_cores)

net = jones_2009_model()
net.clear_connectivity()

data_path = '../../data/hnn_rc'
save_path = '../../data/hnn_rc/validation_sims'
temp_path = '../../data/hnn_rc/temp'

with open(f'{data_path}/posteriors/hnn_rc_posterior_dicts.pkl', 'rb') as output_file:
    posterior_state_dicts = dill.load(output_file)
with open(f'{data_path}/sbi_sims/prior_dict.pkl', 'rb') as output_file:
    prior_dict = dill.load(output_file)
with open(f'{data_path}/sbi_sims/sim_metadata.pkl', 'rb') as output_file:
    sim_metadata = dill.load(output_file)
with open(f'{data_path}/posteriors/hnn_rc_posterior_metadata.pkl', 'rb') as output_file:
    posterior_metadata = dill.load(output_file)
    
dt = sim_metadata['dt'] # Sampling interval used for simulation
tstop = sim_metadata['tstop'] # Sampling interval used for simulation
zero_samples = posterior_metadata['zero_samples']

t_vec = np.linspace(0, tstop, np.round(tstop/dt).astype(int))


prior = UniformPrior(parameters=list(prior_dict.keys()))
n_params = len(prior_dict)
limits = list(prior_dict.values())

# x_orig stores full waveform to be used for embedding
x_orig, theta_orig = np.load(f'{data_path}/sbi_sims/x_sbi.npy'), np.load(f'{data_path}/sbi_sims/theta_sbi.npy')
x_cond, theta_cond = np.load(f'{data_path}/sbi_sims/x_grid.npy'), np.load(f'{data_path}/sbi_sims/theta_grid.npy')

x_orig[:, :zero_samples] = np.repeat(x_orig[:, zero_samples], zero_samples).reshape(x_orig.shape[0], zero_samples)
x_cond[:, :zero_samples] = np.repeat(x_cond[:, zero_samples], zero_samples).reshape(x_cond.shape[0], zero_samples)

load_info = {name: {'x_train': posterior_dict['input_dict']['feature_func'](x_orig), 
                    'x_cond': posterior_dict['input_dict']['feature_func'](x_cond)}
             for name, posterior_dict in posterior_state_dicts.items()}

# Create batch simulation function
def batch(seq, theta_samples, save_path):
    # create simulator object, rescale function transforms (0,1) to range specified in prior_dict
    simulator = partial(simulator_hnn, prior_dict=prior_dict, param_function=hnn_rc_param_function,
                        network_model=net)

    # Create lazy list of tasks
    res_list= []
    for sim_idx in range(len(seq)):
        res = dask.delayed(simulator)(theta_samples[sim_idx,:])
        res_list.append(res)

    # Run tasks
    final_res = dask.compute(*res_list)
    x_list = np.stack([final_res[idx][0] for idx in range(len(seq))])

    x_name = f'{temp_path}/x_val{seq[0]}-{seq[-1]}.npy'
    theta_name = f'{temp_path}/theta_val{seq[0]}-{seq[-1]}.npy'

    np.save(x_name, x_list)
    np.save(theta_name, theta_samples.detach().cpu().numpy())
    

for input_type, posterior_dict in posterior_state_dicts.items():
    if input_type in ['pca5', 'pca10']:
        continue
    state_dict = posterior_dict['posterior']
    n_params = posterior_dict['n_params']
    n_sims = posterior_dict['n_sims']
    input_dict = posterior_dict['input_dict']

    embedding_net =  input_dict['embedding_func'](**input_dict['embedding_dict'])

    posterior = load_posterior(state_dict=state_dict,
                               x_infer=torch.tensor(load_info[input_type]['x_train'][:10,:]).float(),
                               theta_infer=torch.tensor(theta_orig[:10,:]), prior=prior, embedding_net=embedding_net)

    
    samples_list = list()
    for cond_idx in range(x_cond.shape[0]):
        if cond_idx % 100 == 0:    
            print(cond_idx, end=' ')
        samples = posterior.sample((10,), x=load_info[input_type]['x_cond'][cond_idx,:])
        samples_list.append(samples)
        
    theta_samples = torch.tensor(np.vstack(samples_list))
    
    num_sims = theta_samples.shape[0]

      # Generate simulations
    seq_list = list()
    for i in range(0, num_sims, step_size):
        print(i)
        if i + step_size < theta_samples.shape[0]:
            seq = list(range(i, i + step_size))
            batch(seq, theta_samples[i:i + step_size, :], temp_path)
        else:
            print(i, theta_samples.shape[0])
            seq = list(range(i, theta_samples.shape[0]))
            batch(seq, theta_samples[i:, :], temp_path)
        seq_list.append(seq)

    # Load simulations into single array, save output, and remove small small files
    x_files = [f'{temp_path}/x_val{seq[0]}-{seq[-1]}.npy' for seq in seq_list]
    theta_files = [f'{temp_path}/theta_val{seq[0]}-{seq[-1]}.npy' for seq in seq_list]
    
    x_final, theta_final = load_prerun_simulations(x_files, theta_files)
    x_name = f'{save_path}/x_{input_type}_validation.npy'
    theta_name = f'{save_path}/theta_{input_type}_validation.npy'
    np.save(x_name, x_final)
    np.save(theta_name, theta_final)

    files = glob.glob(str(temp_path) + '/*')
    for f in files:
        os.remove(f)

os.system('scancel -u ntolley')
