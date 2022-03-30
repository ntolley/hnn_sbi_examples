import numpy as np
from scipy.integrate import odeint
from scipy import signal

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

def run_rc_sim(theta_dict):    
    tstop = 80
    dt = 0.5
    t_vec = np.linspace(0, tstop, np.round(tstop/dt).astype(int))

    amp1 = theta_dict['amp1']
    amp2 = theta_dict['amp2']
    pulse_diff = theta_dict['pulse_diff']

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

def get_dataset_psd(x_raw, fs, max_freq=500):
    """Calculate PSD on observed time series (rows of array)"""
    x_psd = list()
    for idx in range(x_raw.shape[0]):
        f, Pxx_den = signal.welch(x_raw[idx, :], fs)
        x_psd.append(Pxx_den[f<max_freq])
    return np.vstack(np.log(x_psd))