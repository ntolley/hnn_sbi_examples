import sys
sys.path.append('../')
import os
from utils import (hnn_erp_param_function, validate_posterior, start_cluster)
from hnn_core import jones_2009_model

net = jones_2009_model()

nval_sims = 10
data_path = '../../data/hnn_erp'

start_cluster() # reserve resources for HNN simulations
validate_posterior(net, nval_sims, hnn_erp_param_function, data_path)

#os.system('scancel -u ntolley')
