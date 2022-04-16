import numpy as np
import dill
import torch
from utils import (linear_scale_forward, run_rc_sim, UniformPrior, linear_scale_array,
                   get_dataset_psd, get_dataset_peaks, load_posterior)
from sklearn.decomposition import PCA

device = 'cpu'
data_path = '../data/rc_circuit'

with open(f'{data_path}/posteriors/rc_posterior_dicts.pkl', 'rb') as output_file:
    posterior_state_dicts = dill.load(output_file)
with open(f'{data_path}/sbi_sims/prior_dict.pkl', 'rb') as output_file:
    prior_dict = dill.load(output_file)
with open(f'{data_path}/sbi_sims/sim_metadata.pkl', 'rb') as output_file:
    sim_metadata = dill.load(output_file)
with open(f'{data_path}/posteriors/rc_posterior_metadata.pkl', 'rb') as output_file:
    posterior_metadata = dill.load(output_file)
    
dt = sim_metadata['dt'] # Sampling interval used for simulation
tstop = sim_metadata['tstop'] # Sampling interval used for simulation

t_vec = np.linspace(0, tstop, np.round(tstop/dt).astype(int))


prior = UniformPrior(parameters=list(prior_dict.keys()))
n_params = len(prior_dict)
limits = list(prior_dict.values())

# x_orig stores full waveform to be used for embedding
x_orig, theta_orig = np.load(f'{data_path}/sbi_sims/x_sbi.npy'), np.load(f'{data_path}/sbi_sims/theta_sbi.npy')
x_cond, theta_cond = np.load(f'{data_path}/sbi_sims/x_grid.npy'), np.load(f'{data_path}/sbi_sims/theta_grid.npy')

x_orig_peak = get_dataset_peaks(x_orig)
x_cond_peak = get_dataset_peaks(x_cond)


fs = posterior_metadata['fs'] # Frequency defined for PSD
x_orig_psd, f = get_dataset_psd(x_orig, fs=fs)
x_cond_psd, f = get_dataset_psd(x_cond, fs=fs)

pca = PCA(n_components=10, random_state=rng_seed)
pca.fit(x_orig)
x_orig_pca = pca.transform(x_orig)
x_cond_pca = pca.transform(x_cond)

load_info = {'raw_waveform_noise': {'x_train': x_orig, 'x_cond': x_cond},
             'pca_noise': {'x_train': x_orig_pca, 'x_cond': x_cond_pca},
             'peak_noise': {'x_train': x_orig_peak, 'x_cond': x_cond_peak},
             'psd_noise': {'x_train': x_orig_psd, 'x_cond': x_cond_psd}}



for name, posterior_dict in posterior_state_dicts.items():
    state_dict = posterior_dict['posterior']
    n_params = posterior_dict['n_params']
    input_type = posterior_dict['input_type']
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
        samples = posterior.sample((100,), x=load_info[input_type]['x_cond'][cond_idx,:])
        samples_list.append(samples)
        
    theta_samples = np.vstack(samples_list)
    
    num_sims = theta_samples.shape[0]

    v_list = list()
    for sim_idx in range(num_sims):
        if sim_idx % 1000 == 0:
            print(sim_idx, end=' ')
        thetai = theta_samples[sim_idx, :]
        theta_dict = {param_name: param_dict['scale_func'](thetai[idx], param_dict['bounds']) for 
                      idx, (param_name, param_dict) in enumerate(prior_dict.items())}

        v_out = run_rc_sim(theta_dict, tstop=sim_metadata['tstop'], dt=sim_metadata['dt'])
        v_list.append(v_out)

    # Save simulation output
    save_path = '../data/rc_circuit/validation_sims'
    x_sims = np.hstack(v_list).T

    x_name = f'{save_path}/x_{input_type}_validation.npy'
    theta_name = f'{save_path}/theta_{input_type}_validation.npy'
    np.save(x_name, x_sims)
    np.save(theta_name, theta_samples)

