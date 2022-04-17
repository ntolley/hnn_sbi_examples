import torch
import os
import numpy as np

from sbi import inference as sbi_inference
from utils import (linear_scale_forward,
                   UniformPrior, linear_scale_array,
                   get_dataset_psd, get_dataset_peaks)
import pickle
import dill
from sbi import utils as sbi_utils
from sklearn.decomposition import PCA

from numpy.random import default_rng
rng_seed = 123
rng = default_rng(123)

device = 'cpu'

n_sims = 100_000

posterior_dict = dict()
posterior_dict_training_data = dict()

data_path = '../data/rc_circuit'

prior_dict = pickle.load(open(f'{data_path}/sbi_sims/prior_dict.pkl', 'rb'))
sim_metadata = pickle.load(open(f'{data_path}/sbi_sims/sim_metadata.pkl', 'rb'))

prior = UniformPrior(parameters=list(prior_dict.keys()))
n_params = len(prior_dict)
limits = list(prior_dict.values())

# x_orig stores full waveform to be used for embedding
x_orig, theta_orig = np.load(f'{data_path}/sbi_sims/x_sbi.npy'), np.load(f'{data_path}/sbi_sims/theta_sbi.npy')
x_orig, theta_orig = x_orig[:n_sims, :], theta_orig[:n_sims, :]

# Add noise for regularization
noise_amp = 0.1
noise = np.random.random(x_orig.shape) * noise_amp - (noise_amp / 2)
x_orig_noise = x_orig + noise

dt = sim_metadata['dt'] # Sampling interval used for simulation
fs = (1/dt) * 1e3

def psd_peak_func(x_raw, fs):
    x_psd = get_dataset_psd(x_raw, fs=fs, return_freq=False)
    x_peak = get_dataset_peaks(x_raw)
    return np.hstack([x_psd, x_peak])

pca5 = PCA(n_components=5, random_state=rng_seed)
pca5.fit(x_orig_noise)

pca10 = PCA(n_components=10, random_state=rng_seed)
pca10.fit(x_orig_noise)

pca30 = PCA(n_components=30, random_state=rng_seed)
pca30.fit(x_orig_noise)

posterior_metadata = {'rng_seed': rng_seed, 'noise_amp': noise_amp, 'n_sims': n_sims, 'fs': fs}
posterior_metadata_save_label = f'{data_path}/posteriors/rc_posterior_metadata.pkl'
with open(posterior_metadata_save_label, 'wb') as output_file:
        dill.dump(posterior_metadata, output_file)

input_type_list = {'raw_waveform': {
                       'embedding_func': torch.nn.Identity,
                       'embedding_dict': dict(), 'feature_func': torch.nn.Identity},
                   'pca5': {
                       'embedding_func': torch.nn.Identity,
                       'embedding_dict': dict(), 'feature_func': pca5.transform},
                   'pca10': {
                       'embedding_func': torch.nn.Identity,
                       'embedding_dict': dict(), 'feature_func': pca10.transform},
                   'pca30': {
                       'embedding_func': torch.nn.Identity,
                       'embedding_dict': dict(), 'feature_func': pca30.transform},
                   'peak': {
                       'embedding_func': torch.nn.Identity,
                       'embedding_dict': dict(), 'feature_func': get_dataset_peaks},
                   'psd': {
                       'embedding_func': torch.nn.Identity,
                       'embedding_dict': dict(), 'feature_func': partial(get_dataset_psd, fs=fs, return_freq=False)},
                   'psd_peak': {
                       'embedding_func': torch.nn.Identity,
                       'embedding_dict': dict(), 'feature_func': partial(psd_peak_func, fs=fs)}}

# Train a posterior for each input type and save state_dict
for input_type, input_dict in input_type_list.items():
    dict_key = f'rc_{input_type}'
    print(dict_key)

    neural_posterior = sbi_utils.posterior_nn(model='maf', embedding_net=input_dict['embedding_func'](**input_dict['embedding_dict']))
    inference = sbi_inference.SNPE(prior=prior, density_estimator=neural_posterior, show_progress_bars=True, device=device)
    x_train = torch.tensor(input_dict['feature_func'](x_orig_noise)).float()
    theta_train = torch.tensor(theta_orig).float()
    if x_train.dim() == 1:
        x_train= x_train.reshape(-1, 1)

    inference.append_simulations(theta_train, x_train, proposal=prior)

    nn_posterior = inference.train(num_atoms=10, training_batch_size=5000, use_combined_loss=True, discard_prior_samples=True, max_num_epochs=None, show_train_summary=True)

    posterior_dict[dict_key] = {'posterior': nn_posterior.state_dict(),
                                'n_params': n_params,
                                'input_type': input_type,
                                'n_sims': n_sims,
                                'input_dict': input_dict}

    # Save intermediate progress
    posterior_save_label = f'{data_path}/posteriors/rc_posterior_dicts.pkl'
    with open(posterior_save_label, 'wb') as output_file:
        dill.dump(posterior_dict, output_file)
        
os.system('scancel -u ntolley')
