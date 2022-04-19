import numpy as np
from scipy.integrate import odeint
from scipy import signal
import torch

from sbi import utils as sbi_utils
from sbi import analysis as sbi_analysis
from sbi import inference as sbi_inference

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
    
    def __init__(self, prior_dict, param_function, rescale_function, network_model,
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
        rescale_function: function definition
            Function which scales parameter values passed in __call__() to bounds in prior_dict
            i.e linear or log scale.
        network_model: function definiton
            Function defined in network_models.py of hnn_core which builds the desired Network to
            be simulated.
        return_objects: bool
            If true, returns tuple of (Network, Dipole) objects. If False, a preprocessed time series
            of the aggregate current dipole (Dipole.data['agg']) is returned.
        """
        self.dt = 0.5  # Used for faster simulations, default.json uses 0.025 ms
        self.tstop = 350  # ms
        self.prior_dict = prior_dict
        self.param_function = param_function
        self.rescale_function = rescale_function
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
        dpl = simulate_dipole(net, tstop=self.tstop, dt=self.dt, n_trials=1, postproc=True)

        # get the signal output, downsample by factor of 50
        x = torch.tensor(dpl[0].copy().smooth(20).data['agg'][::3], dtype=torch.float32)
        
        if self.return_objects:
            return net, dpl
        else:
            del net, dpl
            return x      

def simulator_hnn(theta, prior_dict, param_function, rescale_function, network_model=jones_2009_model,
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
    rescale_function: function definition
        Function which scales parameter values passed in __call__() to bounds in prior_dict
        i.e linear or log scale.
    network_model: function definiton
        Function defined in network_models.py of hnn_core which builds the desired Network to
        be simulated.
    return_objects: bool
        If true, returns tuple of (Network, Dipole) objects. If False, a preprocessed time series
        of the aggregate current dipole (Dipole.data['agg']) is returned.
    """
    
    # Convert rescale function to list if different for each parameter
    if isinstance(rescale_function, list):
        if theta.ndim == 1:
            assert len(rescale_function) == theta.shape[0]
        else:
            assert len(rescale_function) == theta.shape[1]
    elif callable(rescale_function):
        rescale_function = [rescale_function for _ in range(len(theta))]
    else:
        raise TypeError

    # create simulator
    hnn = HNNSimulator(prior_dict, param_function, rescale_function, network_model, return_objects)

    # handle when just one theta
    if theta.ndim == 1:
        return simulator_hnn(theta.view(1, -1), prior_dict, param_function, rescale_function,
                             return_objects=return_objects, network_model=network_model)

    # loop through different values of theta
    x = list()
    for idx, thetai in enumerate(theta):
        theta_dict = {param_name: rescale_function[idx](thetai[idx].numpy(), bounds) for 
                      idx, (param_name, bounds) in enumerate(prior_dict.items())}
        
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
        'prox', mu=prox_start, sigma=0.0, numspikes=1, weights_ampa=weights_ampa_dist, location='proximal',
        synaptic_delays=synaptic_delays)


    net.add_evoked_drive(
        'dist', mu=dist_start, sigma=0.0, numspikes=1, weights_ampa=weights_ampa_dist, location='distal',
        synaptic_delays=synaptic_delays)