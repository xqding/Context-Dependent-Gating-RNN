import numpy as np
from parameters import *
import model
import sys, os
import pickle


def try_model(save_fn,gpu_id):

    try:
        # Run model
        model.main(save_fn, gpu_id)
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt')

###############################################################################
###############################################################################
###############################################################################


# Second argument will select the GPU to use
# Don't enter a second argument if you want TensorFlow to select the GPU/CPU
try:
    gpu_id = sys.argv[1]
    print('Selecting GPU ', gpu_id)
except:
    gpu_id = None

omega_c = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])

"""
# BIO - R Learning - XdG
for j in range(5):
    for i in [3]:
        update_parameters({'omega_c': omega_c[i], 'gating_type': 'XdG', 'architecture': 'BIO', 'training_method': 'SL',\
            'n_train_batches': 8000, 'synapse_config': 'std_stf', 'exc_inh_prop': 0.8, 'mask_duration': 0,\
            'learning_rate': 5e-4, 'val_cost': 0.01, 'entropy_cost': 1e-5, 'omega_xi': 0.01,'n_hidden': 256})
        save_fn = 'BIO_SL_XdG_omega' + str(i) + '_v' + str(j) + '.pkl'
        try_model(save_fn, gpu_id)
quit()



# LSTM - Reinforcement Learning - XdG
for j in range(2,5):
    for i in [3]:
        update_parameters({'omega_c': omega_c[i], 'gating_type': 'XdG', 'architecture': 'LSTM', 'training_method': 'RL',\
            'n_train_batches': 50000, 'learning_rate': 5e-4, 'val_cost': 0.01, 'entropy_cost': 1e-4, 'omega_xi': 0.01})
        save_fn = 'LSTM_RL_XdG_xi01_sgd_omega' + str(i) + '_v' + str(j) + '.pkl'
        try_model(save_fn, gpu_id)
quit()



# LSTM - Supervised Learning - XdG
for j in range(3, 5):
    for i in [7]:
        update_parameters({'omega_c': omega_c[i], 'gating_type': 'XdG', 'architecture': 'LSTM', 'training_method': 'SL',\
            'n_train_batches': 6000})
        save_fn = 'LSTM_SL_XdG_omega' + str(i) + '_v' + str(j) + '.pkl'
        try_model(save_fn, gpu_id)
quit()
"""

# LSTM - Supervised Learning - No gating
for j in [1,2,3,4]:
    for i in [7]:
        update_parameters({'omega_c': omega_c[i], 'gating_type': None, 'architecture': 'LSTM', 'training_method': 'SL',\
            'n_train_batches': 6000, 'include_rule_signal': True, 'num_rule_tuned': 20, 'gate_pct': 0.5})
        save_fn = 'LSTM_SL_no_gating_rule_signal_omega' + str(i) + '_v' + str(j) + '.pkl'
        try_model(save_fn, gpu_id)



# LSTM - Supervised Learning - No gating
for j in range(1,5):
    for i in [5,7]:
        update_parameters({'omega_c': omega_c[i], 'gating_type': 'partial', 'architecture': 'LSTM', 'training_method': 'SL',\
            'n_train_batches': 6000, 'gate_pct': 0.5})
        save_fn = 'LSTM_SL_partial_gating_omega' + str(i) + '_v' + str(j) + '.pkl'
        try_model(save_fn, gpu_id)
