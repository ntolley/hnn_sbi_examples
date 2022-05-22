import sys
sys.path.append('../../')
import os
import numpy as np
import dill
import torch
from functools import partial
from scipy.stats import wasserstein_distance
from utils import (linear_scale_forward, log_scale_forward, UniformPrior,
                   simulator_hnn, hnn_erp_param_function, load_prerun_simulations,
                   get_dataset_psd, get_dataset_peaks, load_posterior)
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
sns.set_style("white")

device = 'cpu'


sim_type = 'hnn_erp'
data_path = f'../../../data/{sim_type}'

with open(f'{data_path}/posteriors/posterior_dicts.pkl', 'rb') as output_file:
    posterior_state_dicts = dill.load(output_file)
with open(f'{data_path}/sbi_sims/prior_dict.pkl', 'rb') as output_file:
    prior_dict = dill.load(output_file)
with open(f'{data_path}/sbi_sims/sim_metadata.pkl', 'rb') as output_file:
    sim_metadata = dill.load(output_file)
with open(f'{data_path}/posteriors/posterior_metadata.pkl', 'rb') as output_file:
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

# Parameter recovery plots
plot_labels = ['Distal Inh dist', 'Proximal Inh dist', 'Distal Exc dist', 'Proximal Exc dist']
param_labels = ['Distal Inh (log g)', 'Proximal Inh (log g)', 'Distal Exc (log g)', 'Proximal Exc (log g)']
all_bounds = [param_dict['bounds'] for param_dict in prior_dict.values()]

labelsize=16
for input_type, posterior_dict in posterior_state_dicts.items():
    print(input_type)
    
    theta_val = np.load(f'{data_path}/sbi_sims/theta_{input_type}_validation.npy')

    dist_list = list()
    for cond_idx in range(theta_cond.shape[0]):
        start_idx, stop_idx = cond_idx*10, (cond_idx+1)*10
        dist = [wasserstein_distance(theta_val[start_idx:stop_idx, param_idx], [theta_cond[cond_idx,param_idx]]) for
                param_idx in range(theta_cond.shape[1])]
        dist_list.append(dist)
    dist_array = np.array(dist_list)

    plt.figure(figsize=(17,4))
    for plot_idx in range(4):
        plt.subplot(1,4,plot_idx+1)
        xticks = np.round(np.linspace(all_bounds[3][0], all_bounds[3][1], 10), decimals=2)
        yticks = np.round(np.linspace(all_bounds[0][0], all_bounds[0][1], 10), decimals=2)
        sns.heatmap(dist_array[:,plot_idx].reshape(10,10,10,10)[:,5,5,:], vmin=0, vmax=0.3,
                    xticklabels=xticks, yticklabels=yticks)
        plt.title(plot_labels[plot_idx])
        plt.xlabel(param_labels[2], fontsize=labelsize)
        plt.ylabel(param_labels[0], fontsize=labelsize)
    plt.tight_layout()
    plt.savefig(f'../../../figures/{sim_type}/wasserstein_{sim_type}_{input_type}.svg')
    plt.close()
    
    
# PPC plots

for input_type, posterior_dict in posterior_state_dicts.items():
    print(input_type)

    x_val = np.load(f'{data_path}/sbi_sims/x_{input_type}_validation.npy')
    x_val[:, :zero_samples] = np.zeros(x_val[:, :zero_samples].shape)

    theta_val = np.load(f'{data_path}/sbi_sims/theta_{input_type}_validation.npy')

    dist_list = list()
    for cond_idx in range(theta_cond.shape[0]):
        start_idx, stop_idx = cond_idx*10, (cond_idx+1)*10
        dist = np.sqrt(np.mean(np.square(x_val[start_idx:stop_idx,:] - np.tile(x_cond[cond_idx,:], 10).reshape(10,-1))))
        dist_list.append(dist)
    dist_array = np.array(dist_list)

    plt.figure(figsize=(5,5))
    xticks = np.round(np.linspace(all_bounds[3][0], all_bounds[3][1], 10), decimals=2)
    yticks = np.round(np.linspace(all_bounds[0][0], all_bounds[0][1], 10), decimals=2)
    sns.heatmap(dist_array.reshape(10,10,10,10)[:,5,5,:], vmin=0, vmax=0.075,
                xticklabels=xticks, yticklabels=yticks, cmap='viridis')
    plt.title(input_type)
    plt.xlabel(param_labels[3])
    plt.ylabel(param_labels[0])
    plt.tight_layout()
    plt.savefig(f'../../../figures/{sim_type}/ppc_{sim_type}_{input_type}.svg')
    plt.close()
