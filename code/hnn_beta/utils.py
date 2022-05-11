import numpy as np
from scipy.integrate import odeint
from scipy import signal
import torch
import glob

from sbi import utils as sbi_utils
from sbi import analysis as sbi_analysis
from sbi import inference as sbi_inference

from hnn_core import jones_2009_model, simulate_dipole, pick_connection
rng = np.random.default_rng(123)

device = 'cpu'

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
    
    def __init__(self, prior_dict, param_function, network_model,
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
        return_objects: bool
            If true, returns tuple of (Network, Dipole) objects. If False, a preprocessed time series
            of the aggregate current dipole (Dipole.data['agg']) is returned.
        """
        self.dt = 0.5  # Used for faster simulations, default.json uses 0.025 ms
        self.tstop = 500  # ms
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

        # get the signal output, downsample
        x_l2 = torch.tensor(dpl[0].copy().smooth(20).data['L2'], dtype=torch.float32)
        x_l5 = torch.tensor(dpl[0].copy().smooth(20).data['L5'], dtype=torch.float32)
        x_agg = torch.tensor(dpl[0].copy().smooth(20).data['agg'], dtype=torch.float32)
        
        x = torch.stack([x_agg, x_l2, x_l5])
        
        if self.return_objects:
            return net, dpl
        else:
            del net, dpl
            return x      

def simulator_hnn(theta, prior_dict, param_function, network_model=jones_2009_model,
                  return_objects=False):
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
    return_objects: bool
        If true, returns tuple of (Network, Dipole) objects. If False, a preprocessed time series
        of the aggregate current dipole (Dipole.data['agg']) is returned.
    """

    # create simulator
    hnn = HNNSimulator(prior_dict, param_function, network_model, return_objects)

    # handle when just one theta
    if theta.ndim == 1:
        return simulator_hnn(theta.view(1, -1), prior_dict, param_function,
                             return_objects=return_objects, network_model=network_model)

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
    
def hnn_beta_param_function(net, theta_dict, rng=rng):
    beta_start = 200.0
    prox_seed = rng.integers(1000)
    dist_seed = rng.integers(1000)

    # Distal Drive
    weights_ampa_d1 = {'L2_basket': 0.8e-6, 'L2_pyramidal': 0.4e-6,
                       'L5_pyramidal': theta_dict['dist_exc']}
    syn_delays_d1 = {'L2_basket': 0.0, 'L2_pyramidal': 0.0,
                     'L5_pyramidal': 0.0}
    net.add_bursty_drive(
        'beta_dist', tstart=beta_start, tstart_std=0., tstop=beta_start + 50.,
        burst_rate=1., burst_std=theta_dict['dist_var'], numspikes=2, spike_isi=10,
        n_drive_cells=10, location='distal', weights_ampa=weights_ampa_d1,
        synaptic_delays=syn_delays_d1, event_seed=dist_seed)

    # Proximal Drive
    weights_ampa_p1 = {'L2_basket': 0.4e-6, 'L2_pyramidal': 0.2e-6,
                       'L5_basket': 0.4e-6, 'L5_pyramidal': theta_dict['prox_exc']}
    syn_delays_p1 = {'L2_basket': 0.0, 'L2_pyramidal': 0.0,
                     'L5_basket': 0.0, 'L5_pyramidal': 0.0}

    net.add_bursty_drive(
        'beta_prox', tstart=beta_start, tstart_std=0., tstop=beta_start + 50.,
        burst_rate=1., burst_std=theta_dict['prox_var'], numspikes=2, spike_isi=10,
        n_drive_cells=10, location='proximal', weights_ampa=weights_ampa_p1,
        synaptic_delays=syn_delays_p1, event_seed=prox_seed)
   
        
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