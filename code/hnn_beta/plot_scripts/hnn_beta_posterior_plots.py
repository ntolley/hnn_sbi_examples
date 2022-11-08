import sys
sys.path.append('../../')
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from sbi import inference as sbi_inference
from utils import (linear_scale_forward, hnn_beta_param_function, UniformPrior, linear_scale_array,
                   get_dataset_psd, get_dataset_peaks, load_posterior, simulator_hnn)
from hnn_core import jones_2009_model
from functools import partial
import dill
from sbi import utils as sbi_utils
import pandas as pd
from sklearn.decomposition import PCA
rng_seed = 123

sns.set()
sns.set_style("white")

device = 'cpu'

sim_type = 'hnn_beta'
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


# Parameter bounds
for param_name, param_dict in prior_dict.items():
    print(f'{param_name}:{param_dict["bounds"]}', end=' ')
    
    
net = jones_2009_model()

simulator = partial(simulator_hnn, prior_dict=prior_dict, param_function=hnn_beta_param_function,
                    network_model=net, tstop=tstop)


# Values on [0,1] mapped to range of bounds defined in prior_dict
theta_cond_list = [np.array([0.25, 0.5, 0.99, 0.99]), np.array([0.75, 0.5, 0.75, 0.75])]
for plot_idx, theta_cond in enumerate(theta_cond_list):

    theta_dict = {param_name: param_dict['rescale_function'](theta_cond[idx], param_dict['bounds']) for 
                  idx, (param_name, param_dict) in enumerate(prior_dict.items())}

    x_cond = simulator(torch.tensor(theta_cond).float()).numpy()
    x_cond[:, :zero_samples] = np.repeat(x_cond[:, zero_samples], zero_samples).reshape(x_cond.shape[0], zero_samples)

    print(theta_dict)


    # Plot conditioning vector
    plt.figure()
    t_vec = np.linspace(0, tstop, x_cond.shape[1])
    plt.plot(t_vec, x_cond.squeeze())
    plt.xlabel('Time (ms)')
    plt.ylabel('Dipole (nAm)')
    plt.savefig(f'../../../figures/{sim_type}/posterior_cond_{sim_type}_{plot_idx}.svg')

    prior = UniformPrior(parameters=list(prior_dict.keys()))
    n_params = len(prior_dict)
    limits = list(prior_dict.values())

    # Load posterior
    # x_orig stores full waveform to be used for embedding
    x_orig, theta_orig = np.load(f'{data_path}/sbi_sims/x_sbi.npy'), np.load(f'{data_path}/sbi_sims/theta_sbi.npy')
    x_orig[:, :zero_samples] = np.zeros(x_orig[:, :zero_samples].shape)

    x_orig_peak = get_dataset_peaks(x_orig)
    x_cond_peak = get_dataset_peaks(x_cond.T)

    fs = posterior_metadata['fs'] # Frequency defined for PSD
    x_orig_psd, f = get_dataset_psd(x_orig, fs=fs)
    x_cond_psd, f = get_dataset_psd(x_cond, fs=fs)

    load_info = {name: {'x_train': posterior_dict['input_dict']['feature_func'](x_orig), 
                        'x_cond': posterior_dict['input_dict']['feature_func'](x_cond)}
                 for name, posterior_dict in posterior_state_dicts.items()}


    name_idx = 0 # Pick posterior to load
    for input_type, posterior_dict in posterior_state_dicts.items():
        state_dict = posterior_dict['posterior']
        n_params = posterior_dict['n_params']
        n_sims = posterior_dict['n_sims']
        input_dict = posterior_dict['input_dict']

        embedding_net =  input_dict['embedding_func'](**input_dict['embedding_dict'])

        posterior = load_posterior(state_dict=state_dict,
                                   x_infer=torch.tensor(load_info[input_type]['x_train'][:10,:]).float(),
                                   theta_infer=torch.tensor(theta_orig[:10,:]), prior=prior, embedding_net=embedding_net)

        print(f'Conditioning Feature: {input_type}')
        num_samples = 1000

        all_labels = ['Distal Var (ms)', 'Proximal Var (ms)', 'Distal log(g)', 'Proximal log(g)']
        all_bounds = [param_dict['bounds'] for param_dict in prior_dict.values()]

        samples = posterior.sample((num_samples,), x=torch.tensor(load_info[input_type]['x_cond'].squeeze()))
        samples_transformed = linear_scale_array(samples.numpy(), all_bounds)

        theta_cond_transformed = linear_scale_array(theta_cond.reshape(1,-1), all_bounds)[0]

        df_dict = {name: samples_transformed[:, idx] for idx, name in enumerate(all_labels)}

        df = pd.DataFrame(df_dict)

        g = sns.PairGrid(df,  diag_sharey=False)
        g.map_lower(sns.scatterplot)
        g.map_upper(sns.kdeplot, fill=True)
        g.map_diag(sns.kdeplot, fill=True)

        for idx in range(4):
            g.axes[idx, idx].axvline(theta_cond_transformed[idx], color='r', linewidth=4) 
            g.axes[idx, idx].set_xlim(all_bounds[idx])
            g.axes[idx, idx].set_ylim(all_bounds[idx])

        for idx1 in range(4):
            for idx2 in range(4):
                g.axes[idx1, idx2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                g.axes[idx1, idx2].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.tight_layout()
        plt.savefig(f'../../../figures/{sim_type}/posterior_scatter_{sim_type}_{input_type}_{plot_idx}.svg')
        plt.close()