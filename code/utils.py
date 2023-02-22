import numpy as np
import dill
import os
from scipy.integrate import odeint
from scipy import signal
from scipy.stats import wasserstein_distance
import torch
import glob
from functools import partial
from dask_jobqueue import SLURMCluster
import dask
from distributed import Client
from sbi import utils as sbi_utils
from sbi import analysis as sbi_analysis
from sbi import inference as sbi_inference
from sklearn.decomposition import PCA
import scipy

from hnn_core import jones_2009_model, simulate_dipole, pick_connection
rng_seed = 123
rng = np.random.default_rng(rng_seed)
torch.manual_seed(rng_seed)
np.random.seed(rng_seed)

device = 'cpu'
num_cores = 256

def run_hnn_sim(net, param_function, prior_dict, theta_samples, tstop, save_path, save_suffix):
    """Run parallel HNN simulations using Dask distributed interface
    
    Parameters
    ----------
    net: Network object
    
    param_function: function definition
        Function which accepts theta_dict and updates simulation parameters 
    prior_dict: dict 
        Dictionary storing information to map uniform sampled parameters to prior distribution.
        Form of {'param_name': {'bounds': (lower_bound, upper_bound), 'scale_func': callable}}.
    theta_samples: array-like
        Unscaled paramter values in range of (0,1) sampled from prior distribution
    tstop: int
        Simulation stop time (ms)
    save_path: str
        Location to store simulations. Must have subdirectories 'sbi_sims/' and 'temp/'
    save_suffix: str
        Name appended to end of output files
    """
    
    # create simulator object, rescale function transforms (0,1) to range specified in prior_dict    
    simulator = partial(simulator_hnn, prior_dict=prior_dict, param_function=param_function,
                        network_model=net, tstop=tstop)
    # Generate simulations
    seq_list = list()
    num_sims = theta_samples.shape[0]
    step_size = num_cores
    
    for i in range(0, num_sims, step_size):
        seq = list(range(i, i + step_size))
        if i + step_size < theta_samples.shape[0]:
            batch(simulator, seq, theta_samples[i:i + step_size, :], save_path)
        else:
            seq = list(range(i, theta_samples.shape[0]))
            batch(simulator, seq, theta_samples[i:, :], save_path)
        seq_list.append(seq)
        
    # Load simulations into single array, save output, and remove small small files
    x_files = [f'{save_path}/temp/x_temp{seq[0]}-{seq[-1]}.npy' for seq in seq_list]
    theta_files = [f'{save_path}/temp/theta_temp{seq[0]}-{seq[-1]}.npy' for seq in seq_list]

    x_orig, theta_orig = load_prerun_simulations(x_files, theta_files)
    x_name = f'{save_path}/sbi_sims/x_{save_suffix}.npy'
    theta_name = f'{save_path}/sbi_sims/theta_{save_suffix}.npy'
    np.save(x_name, x_orig)
    np.save(theta_name, theta_orig)

    files = glob.glob(str(save_path) + '/temp/*')
    for f in files:
        os.remove(f)
        
def run_rc_sim(prior_dict, theta_samples, tstop, save_path, save_suffix):
    """Run RC circuit simulations
    
    Parameters
    ----------
    prior_dict: dict 
        Dictionary storing information to map uniform sampled parameters to prior distribution.
        Form of {'param_name': {'bounds': (lower_bound, upper_bound), 'scale_func': callable}}.
    theta_samples: array-like
        Unscaled paramter values in range of (0,1) sampled from prior distribution
    tstop: int
        Simulation stop time (ms)
    save_path: str
        Location to store simulations. Must have subdirectories 'sbi_sims/' and 'temp/'
    save_suffix: str
        Name appended to end of output files
    """
    # Generate simulations
    num_sims = theta_samples.shape[0]
    step_size = num_cores

    v_list = list()
    for sim_idx in range(num_sims):
        if sim_idx % 1000 == 0:
            print(sim_idx, end=' ')
     
        v_out = simulator_rc(theta_samples[sim_idx, :].squeeze(), prior_dict, tstop=tstop)
        v_list.append(v_out)
        
    # Save simulation output
    x_sims = np.hstack(v_list).T
    theta_sims = theta_samples.numpy()

    x_name = f'{save_path}/sbi_sims/x_{save_suffix}.npy'
    theta_name = f'{save_path}/sbi_sims/theta_{save_suffix}.npy'
    np.save(x_name, x_sims)
    np.save(theta_name, theta_sims)     
        
def start_cluster():
    """Reserve SLURM resources using Dask Distributed interface"""
     # Set up cluster and reserve resources
    cluster = SLURMCluster(
        cores=32, processes=32, queue='compute', memory="256GB", walltime="10:00:00",
        job_extra=['-A csd403', '--nodes=1'], log_directory=os.getcwd() + '/slurm_out')

    client = Client(cluster)
    client.upload_file('../utils.py')
    print(client.dashboard_link)
    
    client.cluster.scale(num_cores)
        
def train_posterior(data_path, ntrain_sims, x_noise_amp, theta_noise_amp, window_samples):
    """Train sbi posterior distribution"""
    posterior_dict = dict()
    posterior_dict_training_data = dict()


    prior_dict = dill.load(open(f'{data_path}/sbi_sims/prior_dict.pkl', 'rb'))
    sim_metadata = dill.load(open(f'{data_path}/sbi_sims/sim_metadata.pkl', 'rb'))

    prior = UniformPrior(parameters=list(prior_dict.keys()))
    n_params = len(prior_dict)
    limits = list(prior_dict.values())

    # x_orig stores full waveform to be used for embedding
    x_orig, theta_orig = np.load(f'{data_path}/sbi_sims/x_sbi.npy'), np.load(f'{data_path}/sbi_sims/theta_sbi.npy')
    x_orig, theta_orig = x_orig[:ntrain_sims, window_samples[0]:window_samples[1]], theta_orig[:ntrain_sims, :]


    # Add noise for regularization
    x_noise = rng.normal(loc=0.0, scale=x_noise_amp, size=x_orig.shape)
    x_orig_noise = x_orig + x_noise
    
    theta_noise = rng.normal(loc=0.0, scale=theta_noise_amp, size=theta_orig.shape)
    theta_orig_noise = theta_orig + theta_noise

    dt = sim_metadata['dt'] # Sampling interval used for simulation
    fs = (1/dt) * 1e3

    pca4 = PCA(n_components=4, random_state=rng_seed)
    pca4.fit(x_orig_noise)
    
    pca30 = PCA(n_components=30, random_state=rng_seed)
    pca30.fit(x_orig_noise)

    posterior_metadata = {'rng_seed': rng_seed, 'x_noise_amp': x_noise_amp, 'theta_noise_amp': theta_noise_amp,
                          'ntrain_sims': ntrain_sims, 'fs': fs, 'window_samples': window_samples}
    posterior_metadata_save_label = f'{data_path}/posteriors/posterior_metadata.pkl'
    with open(posterior_metadata_save_label, 'wb') as output_file:
            dill.dump(posterior_metadata, output_file)

    input_type_list = {#'raw_waveform': {
                       #    'embedding_func': torch.nn.Identity,
                       #    'embedding_dict': dict(), 'feature_func': torch.nn.Identity()},
        
                       'pca4': {
                           'embedding_func': torch.nn.Identity,
                           'embedding_dict': dict(), 'feature_func': pca4.transform},
                       'pca30': {
                           'embedding_func': torch.nn.Identity,
                           'embedding_dict': dict(), 'feature_func': pca30.transform},
                       'peak': {
                           'embedding_func': torch.nn.Identity,
                           'embedding_dict': dict(), 'feature_func': partial(get_dataset_peaks, tstop=sim_metadata['tstop'])},
                       'bandpower': {
                           'embedding_func': torch.nn.Identity,
                           'embedding_dict': dict(), 'feature_func': partial(get_dataset_bandpower, fs=fs)}}
    
                       #'psd': {
                       #    'embedding_func': torch.nn.Identity,
                       #    'embedding_dict': dict(), 'feature_func': partial(get_dataset_psd, fs=fs, return_freq=False)},
                       #'psd_peak': {
                       #    'embedding_func': torch.nn.Identity,
                       #    'embedding_dict': dict(), 'feature_func': partial(psd_peak_func, fs=fs, tstop=sim_metadata['tstop'])}}

    # Train a posterior for each input type and save state_dict
    for input_type, input_dict in input_type_list.items():
        print(input_type)

        neural_posterior = sbi_utils.posterior_nn(model='maf', embedding_net=input_dict['embedding_func'](**input_dict['embedding_dict']))
        inference = sbi_inference.SNPE(prior=prior, density_estimator=neural_posterior, show_progress_bars=True, device=device)
        x_train = torch.tensor(input_dict['feature_func'](x_orig_noise)).float()
        theta_train = torch.tensor(theta_orig_noise).float()
        if x_train.dim() == 1:
            x_train= x_train.reshape(-1, 1)

        inference.append_simulations(theta_train, x_train, proposal=prior)

        nn_posterior = inference.train(num_atoms=10, training_batch_size=5000, use_combined_loss=True, discard_prior_samples=True, max_num_epochs=None, show_train_summary=True)

        posterior_dict[input_type] = {'posterior': nn_posterior.state_dict(),
                                    'n_params': n_params,
                                    'n_sims': ntrain_sims,
                                    'input_dict': input_dict}

        # Save intermediate progress
        posterior_save_label = f'{data_path}/posteriors/posterior_dicts.pkl'
        with open(posterior_save_label, 'wb') as output_file:
            dill.dump(posterior_dict, output_file)
            
            
def validate_posterior(net, nval_sims, param_function, data_path):
        
    # Open relevant files
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
    window_samples = posterior_metadata['window_samples']


    prior = UniformPrior(parameters=list(prior_dict.keys()))

    # x_orig stores full waveform to be used for embedding
    x_orig, theta_orig = np.load(f'{data_path}/sbi_sims/x_sbi.npy'), np.load(f'{data_path}/sbi_sims/theta_sbi.npy')
    x_cond, theta_cond = np.load(f'{data_path}/sbi_sims/x_grid.npy'), np.load(f'{data_path}/sbi_sims/theta_grid.npy')

    x_orig = x_orig[:, window_samples[0]:window_samples[1]]
    x_cond = x_cond[:, window_samples[0]:window_samples[1]]

    load_info = {name: {'x_train': posterior_dict['input_dict']['feature_func'](x_orig), 
                        'x_cond': posterior_dict['input_dict']['feature_func'](x_cond)}
                 for name, posterior_dict in posterior_state_dicts.items()}


    for input_type, posterior_dict in posterior_state_dicts.items():
        state_dict = posterior_dict['posterior']
        input_dict = posterior_dict['input_dict']
        embedding_net =  input_dict['embedding_func'](**input_dict['embedding_dict'])
        
        posterior = load_posterior(state_dict=state_dict,
                                   x_infer=torch.tensor(load_info[input_type]['x_train'][:10,:]).float(),
                                   theta_infer=torch.tensor(theta_orig[:10,:]), prior=prior, embedding_net=embedding_net)


        samples_list = list()
        for cond_idx in range(x_cond.shape[0]):
            if cond_idx % 100 == 0:    
                print(cond_idx, end=' ')
            samples = posterior.sample((nval_sims,), x=load_info[input_type]['x_cond'][cond_idx,:])
            samples_list.append(samples)

        theta_samples = torch.tensor(np.vstack(samples_list))

        save_suffix = f'{input_type}_validation'
        run_hnn_sim(net=net, param_function=param_function, prior_dict=prior_dict,
                theta_samples=theta_samples, tstop=tstop, save_path=data_path, save_suffix=save_suffix)
    
    
# Temporary hack to run rc validation sims
def validate_rc_posterior(nval_sims, data_path):
        
    # Open relevant files
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
    window_samples = posterior_metadata['window_samples']


    prior = UniformPrior(parameters=list(prior_dict.keys()))

    # x_orig stores full waveform to be used for embedding
    x_orig, theta_orig = np.load(f'{data_path}/sbi_sims/x_sbi.npy'), np.load(f'{data_path}/sbi_sims/theta_sbi.npy')
    x_cond, theta_cond = np.load(f'{data_path}/sbi_sims/x_grid.npy'), np.load(f'{data_path}/sbi_sims/theta_grid.npy')

    x_orig = x_orig[:, window_samples[0]:window_samples[1]]
    x_cond = x_cond[:, window_samples[0]:window_samples[1]]

    load_info = {name: {'x_train': posterior_dict['input_dict']['feature_func'](x_orig), 
                        'x_cond': posterior_dict['input_dict']['feature_func'](x_cond)}
                 for name, posterior_dict in posterior_state_dicts.items()}


    for input_type, posterior_dict in posterior_state_dicts.items():
        state_dict = posterior_dict['posterior']
        input_dict = posterior_dict['input_dict']
        embedding_net =  input_dict['embedding_func'](**input_dict['embedding_dict'])
        
        posterior = load_posterior(state_dict=state_dict,
                                   x_infer=torch.tensor(load_info[input_type]['x_train'][:10,:]).float(),
                                   theta_infer=torch.tensor(theta_orig[:10,:]), prior=prior, embedding_net=embedding_net)


        samples_list = list()
        for cond_idx in range(x_cond.shape[0]):
            if cond_idx % 100 == 0:    
                print(cond_idx, end=' ')
            samples = posterior.sample((nval_sims,), x=load_info[input_type]['x_cond'][cond_idx,:])
            samples_list.append(samples)

        theta_samples = torch.tensor(np.vstack(samples_list))

        save_suffix = f'{input_type}_validation'
        run_rc_sim(prior_dict, theta_samples, tstop, data_path, save_suffix)
        

# Create batch simulation function
def batch(simulator, seq, theta_samples, save_path):
    print(f'Sim Idx: {(seq[0], seq[-1])}')
    # Create lazy list of tasks
    res_list = list()
    for sim_idx in range(len(seq)):
        res = dask.delayed(simulator)(theta_samples[sim_idx,:])
        res_list.append(res)

    # Run tasks
    final_res = dask.compute(*res_list)
    x_list = np.stack([final_res[idx][0] for idx in range(len(seq))])

    x_name = f'{save_path}/temp/x_temp{seq[0]}-{seq[-1]}.npy'
    theta_name = f'{save_path}/temp/theta_temp{seq[0]}-{seq[-1]}.npy'

    np.save(x_name, x_list)
    np.save(theta_name, theta_samples.detach().cpu().numpy())

def linear_scale_forward(value, bounds, constrain_value=True):
    """Scale value in range (0,1) to range bounds"""
    if constrain_value:
        assert np.all(value >= 0.0) and np.all(value <= 1.0)
        
    assert isinstance(bounds, tuple)
    assert bounds[0] < bounds[1]
    
    return (bounds[0] + (value * (bounds[1] - bounds[0]))).astype(float)

def linear_scale_array(value, bounds, constrain_value=True):
    """Scale columns of array according to bounds"""
    assert value.shape[1] == len(bounds)
    return np.vstack(
        [linear_scale_forward(value[:, idx], bounds[idx], constrain_value) for 
         idx in range(len(bounds))]).T

def log_scale_forward(value, bounds, constrain_value=True):
    """log scale value in range (0,1) to range bounds in base 10"""
    rescaled_value = linear_scale_forward(value, bounds, constrain_value)
    
    return 10**rescaled_value

def log_scale_array(value, bounds, constrain_value=True):
    """log scale columns of array according to bounds in base 10"""
    assert value.shape[1] == len(bounds)
    return np.vstack(
        [log_scale_forward(value[:, idx], bounds[idx], constrain_value) for 
         idx in range(len(bounds))]).T

def simulator_rc(theta, prior_dict, tstop):  
    dt = 0.5
    t_vec = np.linspace(0, tstop, np.round(tstop/dt).astype(int))
    theta_dict = {param_name: param_dict['scale_func'](theta[param_idx].numpy(), param_dict['bounds']) for 
                      param_idx, (param_name, param_dict) in enumerate(prior_dict.items())}
    
    amp1 = theta_dict['amp1']
    amp2 = theta_dict['amp2']
    pulse_diff = theta_dict['latency']

    y0 = 0
    v_out = odeint(dVdt, y0, t_vec, args=(pulse_diff, amp1, amp2), hmax=dt)
    return v_out


def dVdt(V, t, pulse_diff, amp1, amp2):
    """RC Circuit model of passive neuron
    amp1 and amp2 correspond to magnitude of current injection"""
    
    E = 0
    R = 1
    C = 6
    
    pulse_width = 20
    
    # Current for pulse 1
    i1_start = 200
    i1_stop = i1_start + pulse_width
    
    i1 = float(np.logical_and(t > i1_start, t < i1_stop)) * amp1

    # Current for pulse 2 (negative pulse)
    i2_start = i1_start + pulse_diff
    i2_stop = i2_start + pulse_width
    
    i2 = float(np.logical_and(t > i2_start, t < i2_stop)) * amp2 * -1
    
    I = i1 + i2
        
    return (((E - V) / R) + I) / C

def bandpower(x, fs, fmin, fmax):
    f, Pxx = scipy.signal.periodogram(x, fs=fs)
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

# Bands freq citation: https://www.frontiersin.org/articles/10.3389/fnhum.2020.00089/full
def get_dataset_bandpower(x, fs):
    freq_band_list = [(0,13), (13,30), (30,50), (50,80)]
    
    x_bandpower_list = list()
    for idx in range(x.shape[0]):
        x_bandpower = np.array([bandpower(x[idx,:], fs, freq_band[0], freq_band[1]) for freq_band in freq_band_list])
        x_bandpower_list.append(x_bandpower)
        
    return np.vstack(np.log(x_bandpower_list))

def get_dataset_psd(x_raw, fs, return_freq=True, max_freq=200):
    """Calculate PSD on observed time series (rows of array)"""
    x_psd = list()
    for idx in range(x_raw.shape[0]):
        f, Pxx_den = signal.periodogram(x_raw[idx, :], fs)
        x_psd.append(Pxx_den[(f<max_freq)&(f>0)])
    if return_freq:
        return np.vstack(np.log(x_psd)), f[(f<max_freq)&(f>0)]
    else:
        return np.vstack(np.log(x_psd))
    
# Source: https://stackoverflow.com/questions/44547669/python-numpy-equivalent-of-bandpower-from-matlab
def bandpower(x, fs, fmin, fmax):
    f, Pxx = scipy.signal.periodogram(x, fs=fs)
    ind_min = scipy.argmax(f > fmin) - 1
    ind_max = scipy.argmax(f > fmax) - 1
    return scipy.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])


def get_dataset_peaks(x_raw, tstop=500):
    """Return max/min peak amplitude and timing"""
    ts = np.linspace(0, tstop, x_raw.shape[1])

    peak_features = np.vstack(
        [np.max(x_raw,axis=1), ts[np.argmax(x_raw, axis=1)],
         np.min(x_raw,axis=1), ts[np.argmin(x_raw, axis=1)]]).T

    return peak_features

def psd_peak_func(x_raw, fs, tstop):
    x_psd = get_dataset_psd(x_raw, fs=fs, return_freq=False)
    x_peak = get_dataset_peaks(x_raw, tstop=tstop)
    return np.hstack([x_psd, x_peak])

def load_posterior(state_dict, x_infer, theta_infer, prior, embedding_net):
    """Load a pretrained SBI posterior distribution
    Parameters
    ----------
    """
    neural_posterior = sbi_utils.posterior_nn(model='maf', embedding_net=embedding_net)
    inference = sbi_inference.SNPE(prior=prior, density_estimator=neural_posterior, show_progress_bars=True, device=device)
    inference.append_simulations(theta_infer, x_infer, proposal=prior)

    nn_posterior = inference.train(num_atoms=10, training_batch_size=5000, use_combined_loss=True, discard_prior_samples=True, max_num_epochs=2, show_train_summary=False)
    nn_posterior.zero_grad()
    nn_posterior.load_state_dict(state_dict)

    posterior = inference.build_posterior(nn_posterior)
    return posterior

class UniformPrior(sbi_utils.BoxUniform):
    """Prior distribution object that generates uniform sample on range (0,1)"""
    def __init__(self, parameters):
        """
        Parameters
        ----------
        parameters: list of str
            List of parameter names for prior distribution
        """
        self.parameters = parameters
        low = len(parameters)*[0]
        high = len(parameters)*[1]
        super().__init__(low=torch.tensor(low, dtype=torch.float32),
                         high=torch.tensor(high, dtype=torch.float32))
        
        
# __Simulation__
class HNNSimulator:
    """Simulator class to run HNN simulations"""
    
    def __init__(self, prior_dict, param_function, network_model, tstop,
                 return_objects):
        """
        Parameters
        ----------
        prior_dict: dict 
            Dictionary storing parameters to be updated as {name: (lower_bound, upper_bound)}
            where pameter values passed in the __call__() are scaled between the lower and upper
            bounds
        param_function: function definition
            Function which accepts theta_dict and updates simulation parameters
        network_model: function definiton
            Function defined in network_models.py of hnn_core which builds the desired Network to
            be simulated.
        tstop: int
            Simulation stop time (ms)
        return_objects: bool
            If true, returns tuple of (Network, Dipole) objects. If False, a preprocessed time series
            of the aggregate current dipole (Dipole.data['agg']) is returned.
        """
        self.dt = 0.5  # Used for faster simulations, default.json uses 0.025 ms
        self.tstop = tstop  # ms
        self.prior_dict = prior_dict
        self.param_function = param_function
        self.return_objects = return_objects
        self.network_model = network_model

    def __call__(self, theta_dict):
        """
        Parameters
        ----------
        theta_dict: dict
            Dictionary indexing parameter values to be updated. Keys must match those defined
            in prior_dict
            
        Returns: array-like
            Simulated Output
        """        
        assert len(theta_dict) == len(self.prior_dict)
        assert theta_dict.keys() == self.prior_dict.keys()

        # instantiate the network object -- only connectivity params matter
        net = self.network_model.copy()
        
        # Update parameter values from prior dict
        self.param_function(net, theta_dict)

        # simulate dipole over one trial
        dpl = simulate_dipole(net, tstop=self.tstop, dt=self.dt, n_trials=1, postproc=False)

        # get the signal output
        x = torch.tensor(dpl[0].copy().smooth(20).data['agg'], dtype=torch.float32)
        
        if self.return_objects:
            return net, dpl
        else:
            del net, dpl
            return x      

def simulator_hnn(theta, prior_dict, param_function, network_model,
                  tstop, return_objects=False):
    """Helper function to run simulations with HNN class

    Parameters
    ----------
    theta: array-like
        Unscaled paramter values in range of (0,1) sampled from prior distribution
    prior_dict: dict 
        Dictionary storing parameters to be updated as {name: (lower_bound, upper_bound)}
        where pameter values passed in the __call__() are scaled between the lower and upper
        bounds
    param_function: function definition
        Function which accepts theta_dict and updates simulation parameters
    network_model: function definiton
        Function defined in network_models.py of hnn_core which builds the desired Network to
        be simulated.
    tstop: int
        Simulation stop time (ms)
    return_objects: bool
        If true, returns tuple of (Network, Dipole) objects. If False, a preprocessed time series
        of the aggregate current dipole (Dipole.data['agg']) is returned.
        
    Returns
    -------
    x: array-like
        Simulated output
    """

    # create simulator
    hnn = HNNSimulator(prior_dict, param_function, network_model, tstop, return_objects)

    # handle when just one theta
    if theta.ndim == 1:
        return simulator_hnn(theta.view(1, -1), prior_dict, param_function,
                             return_objects=return_objects, network_model=network_model, tstop=tstop)

    # loop through different values of theta
    x = list()
    for sample_idx, thetai in enumerate(theta):
        theta_dict = {param_name: param_dict['rescale_function'](thetai[param_idx].numpy(), param_dict['bounds']) for 
                      param_idx, (param_name, param_dict) in enumerate(prior_dict.items())}
        
        print(theta_dict)
        xi = hnn(theta_dict)
        x.append(xi)

    # Option to return net and dipole objects or just the 
    if return_objects:
        return x
    else:
        x = torch.stack(x)
        return torch.tensor(x, dtype=torch.float32)
    
def hnn_rc_param_function(net, theta_dict):    
    synaptic_delays = {'L5_pyramidal': 0.1}

    #theta_dict = {'prox_weight': 0.001, 'dist_weight': 0.001, 'latency': -20}
    weights_ampa_prox = {'L5_pyramidal': theta_dict['prox_weight']}
    weights_ampa_dist = {'L5_pyramidal': theta_dict['dist_weight']}
    prox_start = 200.0
    dist_start = prox_start + theta_dict['latency']

    net.add_evoked_drive(
        'prox', mu=prox_start, sigma=0.0, numspikes=1, weights_ampa=weights_ampa_prox, location='proximal',
        synaptic_delays=synaptic_delays)


    net.add_evoked_drive(
        'dist', mu=dist_start, sigma=0.0, numspikes=1, weights_ampa=weights_ampa_dist, location='distal',
        synaptic_delays=synaptic_delays)
    
def hnn_beta_param_function(net, theta_dict, rng=rng):
    beta_start = 200.0
    prox_seed = rng.integers(1000)
    dist_seed = rng.integers(1000)

    # Distal Drive
    weights_ampa_d1 = {'L2_basket': 0.8e-4, 'L2_pyramidal': 0.4e-4,
                       'L5_pyramidal': theta_dict['dist_exc']}
    syn_delays_d1 = {'L2_basket': 0.0, 'L2_pyramidal': 0.0,
                     'L5_pyramidal': 0.0}
    net.add_bursty_drive(
        'beta_dist', tstart=beta_start, tstart_std=0., tstop=beta_start + 50.,
        burst_rate=1., burst_std=theta_dict['dist_var'], numspikes=2, spike_isi=10,
        n_drive_cells=10, location='distal', weights_ampa=weights_ampa_d1,
        synaptic_delays=syn_delays_d1, event_seed=dist_seed)

    # Proximal Drive
    weights_ampa_p1 = {'L2_basket': 0.4e-4, 'L2_pyramidal': 0.2e-4,
                       'L5_basket': 0.4e-4, 'L5_pyramidal': theta_dict['prox_exc']}
    syn_delays_p1 = {'L2_basket': 0.0, 'L2_pyramidal': 0.0,
                     'L5_basket': 0.0, 'L5_pyramidal': 0.0}

    net.add_bursty_drive(
        'beta_prox', tstart=beta_start, tstart_std=0., tstop=beta_start + 50.,
        burst_rate=1., burst_std=theta_dict['prox_var'], numspikes=2, spike_isi=10,
        n_drive_cells=10, location='proximal', weights_ampa=weights_ampa_p1,
        synaptic_delays=syn_delays_p1, event_seed=prox_seed)
    

def hnn_erp_param_function(net, theta_dict):
    # Add ERP drives
    n_drive_cells=1
    cell_specific=False

    weights_ampa_d1 = {'L2_basket': 0.006562, 'L2_pyramidal': .000007,
                       'L5_pyramidal': 0.142300}
    weights_nmda_d1 = {'L2_basket': 0.019482, 'L2_pyramidal': 0.004317,
                       'L5_pyramidal': 0.080074}
    synaptic_delays_d1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                          'L5_pyramidal': 0.1}

    weights_ampa_p1 = {'L2_basket': 0.08831, 'L2_pyramidal': 0.01525,
                       'L5_basket': 0.19934, 'L5_pyramidal': 0.00865}
    synaptic_delays_prox = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                            'L5_basket': 1., 'L5_pyramidal': 1.}

    weights_ampa_p2 = {'L2_basket': 0.000003, 'L2_pyramidal': 1.438840,
                       'L5_basket': 0.008958, 'L5_pyramidal': 0.684013}

    net.add_evoked_drive(
        'evdist1', mu=63.53, sigma=3.85, numspikes=1, weights_ampa=weights_ampa_d1,
        weights_nmda=weights_nmda_d1, location='distal', n_drive_cells=n_drive_cells,
        cell_specific=cell_specific, synaptic_delays=synaptic_delays_d1, event_seed=4)

    net.add_evoked_drive(
        'evprox1', mu=26.61, sigma=2.47, numspikes=1, weights_ampa=weights_ampa_p1,
        weights_nmda=None, location='proximal', n_drive_cells=n_drive_cells,
        cell_specific=cell_specific, synaptic_delays=synaptic_delays_prox, event_seed=4)

    net.add_evoked_drive(
        'evprox2', mu=137.12, sigma=8.33, numspikes=1,
        weights_ampa=weights_ampa_p2, location='proximal', n_drive_cells=n_drive_cells,
        cell_specific=cell_specific, synaptic_delays=synaptic_delays_prox, event_seed=4)
    
    # Update connection weights according to theta_dict
    dist_inh_idx = pick_connection(net, src_gids='L2_basket', target_gids='L5_pyramidal', receptor='gabaa', loc='distal')[0]
    net.connectivity[dist_inh_idx]['nc_dict']['A_weight'] = theta_dict['dist_inh']
    
    prox_inh_idx = pick_connection(net, src_gids='L5_basket', target_gids='L5_pyramidal', receptor='gabaa', loc='soma')[0]
    net.connectivity[prox_inh_idx]['nc_dict']['A_weight'] = theta_dict['prox_inh']

    dist_exc_idx = pick_connection(net, src_gids='L2_pyramidal', target_gids='L5_pyramidal', receptor='ampa', loc='distal')[0]
    net.connectivity[dist_exc_idx]['nc_dict']['A_weight'] = theta_dict['dist_exc']
    
    prox_exc_idx = pick_connection(net, src_gids='L5_pyramidal', target_gids='L5_pyramidal', receptor='ampa', loc='proximal')[0]
    net.connectivity[prox_exc_idx]['nc_dict']['A_weight'] = theta_dict['prox_exc']
    
def load_prerun_simulations(x_files, theta_files, downsample=1, save_name=None, save_data=False):
    "Aggregate simulation batches into single array"
    
    print(x_files)
    print(theta_files)
    
    x_all = np.vstack([np.load(x_files[file_idx])[:,::downsample] for file_idx in range(len(x_files))])
    theta_all = np.vstack([np.load(theta_files[file_idx]) for file_idx in range(len(theta_files))])
    
    if save_data and isinstance(save_name, str):
        np.save(save_name + '_x_all.npy', dpl_all)
        np.save(save_name + '_theta_all.npy', theta_all)
    else:
        return x_all, theta_all
    
def get_parameter_recovery(theta_val, theta_cond, n_samples=10):
    """Calculate the PPC using root mean squared error
    Parameters
    ----------
    x_val: array-like
    
    x_cond: array-like
    
    n_samples: int
    
    Returns
    -------
    dist_array: array-like
    """
    
    dist_list = list()
    for cond_idx in range(theta_cond.shape[0]):
        start_idx, stop_idx = cond_idx*n_samples, (cond_idx+1)*n_samples
        dist = [wasserstein_distance(theta_val[start_idx:stop_idx, param_idx], [theta_cond[cond_idx,param_idx]]) for
                param_idx in range(theta_cond.shape[1])]
        dist_list.append(dist)
    dist_array = np.array(dist_list)
    return dist_array

def get_posterior_predictive_check(x_val, x_cond, n_samples=10):
    """Calculate the PPC using root mean squared error
    Parameters
    ----------
    x_val: array-like
    
    x_cond: array-like
    
    n_samples: int
    
    Returns
    -------
    dist_array: array-like
        
    """
    dist_list = list()
    for cond_idx in range(x_cond.shape[0]):
        start_idx, stop_idx = cond_idx*n_samples, (cond_idx+1)*n_samples
        dist = np.sqrt(np.mean(np.square(x_val[start_idx:stop_idx,:] - np.tile(x_cond[cond_idx,:], n_samples).reshape(n_samples,-1))))
        dist_list.append(dist)
    dist_array = np.array(dist_list)
    return dist_array
