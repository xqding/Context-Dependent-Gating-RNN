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
# BIO - Supervised Learning - XdG
for j in range(1,2):
    for i in [4,3,2,5,1,0,6]:
        update_parameters({'omega_c': omega_c[i], 'gating_type': 'XdG', 'architecture': 'BIO', 'training_method': 'SL',\
            'n_train_batches': 6000, 'synapse_config': 'std_stf', 'exc_inh_prop': 0.8, 'mask_duration': 0})
        save_fn = 'BIO_SL_XdG_omega' + str(i) + '_v' + str(j) + '.pkl'
        try_model(save_fn, gpu_id)
quit()

# LSTM - Reinforcement Learning - XdG
for j in range(0,1):
    for i in [3,4,2,5,1,0,6]:
        update_parameters({'omega_c': omega_c[i], 'gating_type': 'XdG', 'architecture': 'LSTM', 'training_method': 'RL',\
            'n_train_batches': 50000, 'learning_rate': 5e-4, 'val_cost': 0.01, 'entropy_cost': 1e-4})
        save_fn = 'LSTM_RL_XdG_omega' + str(i) + '_v' + str(j) + '.pkl'
        try_model(save_fn, gpu_id)
quit()



# LSTM - Supervised Learning - XdG
for j in range(1):
    for i in [4]:
        update_parameters({'omega_c': omega_c[i], 'gating_type': 'XdG', 'architecture': 'LSTM', 'training_method': 'SL',\
            'n_train_batches': 6000})
        save_fn = 'LSTM_SL_XdG_omega' + str(i) + '_v' + str(j) + '.pkl'
        try_model(save_fn, gpu_id)
quit()
"""


# LSTM - Supervised Learning - No gating
for j in range(1):
    for i in [5]:
        update_parameters({'omega_c': omega_c[i], 'gating_type': None, 'architecture': 'LSTM', 'training_method': 'SL',\
            'n_train_batches': 6000})
        save_fn = 'LSTM_SL_no_gating_omega' + str(i) + '_v' + str(j) + '.pkl'
        try_model(save_fn, gpu_id)
