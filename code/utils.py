import numpy as np
import dill
import os
from scipy.integrate import odeint
from scipy import signal
import torch
import glob
from functools import partial
from dask_jobqueue import SLURMCluster
import dask
from distributed import Client
from sbi import utils as sbi_utils
from sbi import analysis as sbi_analysis
from sbi import inference as sbi_inference

from hnn_core import jones_2009_model, simulate_dipole

device = 'cpu'

def run_hnn_sim(net, param_function, prior_dict, tstop, save_path):
    num_sims = 110_0

    # Set up cluster and reserve resources
    cluster = SLURMCluster(
        cores=32, processes=32, queue='compute', memory="256GB", walltime="3:00:00",
        job_extra=['-A csd403', '--nodes=1'], log_directory=os.getcwd() + '/slurm_out')

    client = Client(cluster)
    client.upload_file('../utils.py')
    print(client.dashboard_link)
    
    num_cores = 256
    step_size = num_cores
    client.cluster.scale(num_cores)
    
    # Create uniform prior and sample
    prior = UniformPrior(parameters=list(prior_dict.keys()))
    theta_samples = prior.sample((num_sims,))

    with open(f'{save_path}/sbi_sims/prior_dict.pkl', 'wb') as f:
        dill.dump(prior_dict, f)

    sim_metadata = {'tstop': tstop, 'dt': 0.5}
    with open(f'{save_path}/sbi_sims/sim_metadata.pkl', 'wb') as f:
        dill.dump(sim_metadata, f)
        
    # create simulator object, rescale function transforms (0,1) to range specified in prior_dict    
    simulator = partial(simulator_hnn, prior_dict=prior_dict, param_function=param_function,
                        network_model=net, tstop=tstop)
    # Generate simulations
    seq_list = list()
    for i in range(0, num_sims, step_size):
        print(i)
        seq = list(range(i, i + step_size))
        if i + step_size < theta_samples.shape[0]:
            batch(simulator, seq, theta_samples[i:i + step_size, :], save_path)
        else:
            print(i, theta_samples.shape[0])
            seq = list(range(i, theta_samples.shape[0]))
            batch(simulator, seq, theta_samples[i:, :], save_path)
        seq_list.append(seq)
        
    # Load simulations into single array, save output, and remove small small files
    x_files = [f'{save_path}/temp/x_sbi{seq[0]}-{seq[-1]}.npy' for seq in seq_list]
    theta_files = [f'{save_path}/temp/theta_sbi{seq[0]}-{seq[-1]}.npy' for seq in seq_list]

    x_orig, theta_orig = load_prerun_simulations(x_files, theta_files)
    x_name = f'{save_path}/sbi_sims/x_sbi.npy'
    theta_name = f'{save_path}/sbi_sims/theta_sbi.npy'
    np.save(x_name, x_orig)
    np.save(theta_name, theta_orig)

    files = glob.glob(str(save_path) + '/temp/*')
    for f in files:
        os.remove(f)

# Create batch simulation function
def batch(simulator, seq, theta_samples, save_path):
    # Create lazy list of tasks
    res_list= []
    for sim_idx in range(len(seq)):
        res = dask.delayed(simulator)(theta_samples[sim_idx,:])
        res_list.append(res)

    # Run tasks
    final_res = dask.compute(*res_list)
    x_list = np.stack([final_res[idx][0] for idx in range(len(seq))])

    x_name = f'{save_path}/temp/x_sbi{seq[0]}-{seq[-1]}.npy'
    theta_name = f'{save_path}/temp/theta_sbi{seq[0]}-{seq[-1]}.npy'

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

def run_rc_sim(theta_dict, tstop=80, dt=0.5):    
    t_vec = np.linspace(0, tstop, np.round(tstop/dt).astype(int))

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
    tau = 2
    
    pulse_width = 10
    
    # Current for pulse 1
    i1_start = 30
    i1_stop = i1_start + pulse_width
    
    i1 = float(np.logical_and(t > i1_start, t < i1_stop)) * amp1

    # Current for pulse 2
    i2_start = i1_start + pulse_diff
    i2_stop = i2_start + pulse_width
    i2 = float(np.logical_and(t > i2_start, t < i2_stop)) * amp2
    
    I = i1 + i2
        
    return (E - V + R * I) / tau

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


def get_dataset_peaks(x_raw, tstop=500):
    """Return max/min peak amplitude and timing"""
    ts = np.linspace(0, tstop, x_raw.shape[1])

    peak_features = np.vstack(
        [np.max(x_raw,axis=1), ts[np.argmax(x_raw, axis=1)],
         np.min(x_raw,axis=1), ts[np.argmin(x_raw, axis=1)]]).T

    return peak_features

def load_posterior(state_dict, x_infer, theta_infer, prior, embedding_net):    
    neural_posterior = sbi_utils.posterior_nn(model='maf', embedding_net=embedding_net)
    inference = sbi_inference.SNPE(prior=prior, density_estimator=neural_posterior, show_progress_bars=True, device=device)
    inference.append_simulations(theta_infer, x_infer, proposal=prior)

    nn_posterior = inference.train(num_atoms=10, training_batch_size=5000, use_combined_loss=True, discard_prior_samples=True, max_num_epochs=2, show_train_summary=False)
    nn_posterior.zero_grad()
    nn_posterior.load_state_dict(state_dict)

    posterior = inference.build_posterior(nn_posterior)
    return posterior

class UniformPrior(sbi_utils.BoxUniform):
    def __init__(self, parameters):
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
            in prior_dict.
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