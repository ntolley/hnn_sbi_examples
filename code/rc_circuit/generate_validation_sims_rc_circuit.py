import sys
sys.path.append('../')
import os
from utils import validate_rc_posterior

nval_sims = 10
data_path = '../../data/rc_circuit'

validate_rc_posterior(nval_sims, data_path)
os.system('scancel -u ntolley')
