import numpy as np
from parameters import *
import model
import sys, os
import pickle


def try_model(save_fn):
    # To use a GPU, from command line do: python model.py <gpu_integer_id>
    # To use CPU, just don't put a gpu id: python model.py
    try:
        if len(sys.argv) > 1:
            model.main(save_fn, sys.argv[1])
        else:
            model.main(save_fn)
    except KeyboardInterrupt:
        print('Quit by KeyboardInterrupt.')


def gen_filename_from_specified_params(params, ext = ".pkl"):
    """
        Generate filename from specified dictionary of params.

        Returns:
            - str
    """
    fn = ext
    for key in params.keys():
        fn = f"{key}_{params[key]}_{fn}"
    return fn

###############################################################################
###############################################################################
###############################################################################


omega_c = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])


"""# BIO - R Learning - XdG
for j in range(5):
    for i in [3]:
        update_parameters({'omega_c': omega_c[i], 'gating_type': 'XdG', 'architecture': 'BIO', 'training_method': 'SL',\
            'n_train_batches': 8000, 'synapse_config': 'std_stf', 'exc_inh_prop': 0.8, 'mask_duration': 0,\
            'learning_rate': 5e-4, 'val_cost': 0.01, 'entropy_cost': 1e-5, 'omega_xi': 0.01,'n_hidden': 256})
        save_fn = 'BIO_SL_XdG_omega' + str(i) + '_v' + str(j) + '.pkl'
        try_model(save_fn)
quit()"""

# Self-experimenting net
params_to_update = {'omega_c'              : omega_c[2], 
                   'architecture'          : 'LSTM', 
                   'training_method'       : 'RL',
                   'n_train_batches'       : 100, 
                   'learning_rate'         : 5e-4, 
                   'val_cost'              : 0.01, 
                   'entropy_cost'          : 1e-4, 
                   'omega_xi'              : 0.01,
                   'task'                  : 'selfexp',
                   'stimulus_choice'       : 'dynamic',
                   'multistim_trial_length': 1000}
update_parameters(params_to_update)
save_fn = gen_filename_from_specified_params(params_to_update)
print(f"Running {save_fn}")
try_model(save_fn)


# LSTM - Reinforcement Learning - XdG
for j in range(1):
    for i in [2]:
        update_parameters({'omega_c': omega_c[i], 'gating_type': 'XdG', 'architecture': 'LSTM', 'training_method': 'RL',\
            'n_train_batches': 50000, 'learning_rate': 5e-4, 'val_cost': 0.01, 'entropy_cost': 1e-4, 'omega_xi': 0.01})
        save_fn = 'testing2_updated_stim_LSTM_RL_XdG_xi01_sgd_omega' + str(i) + '_v' + str(j) + '.pkl'
        print('Running {}'.format(save_fn))
        try_model(save_fn)
quit()


"""
# LSTM - Supervised Learning - XdG
for j in range(3, 5):
    for i in [7]:
        update_parameters({'omega_c': omega_c[i], 'gating_type': 'XdG', 'architecture': 'LSTM', 'training_method': 'SL',\
            'n_train_batches': 6000})
        save_fn = 'LSTM_SL_XdG_omega' + str(i) + '_v' + str(j) + '.pkl'
        try_model(save_fn)
quit()
"""

# LSTM - Supervised Learning - No gating
for j in [1,2,3,4]:
    for i in [7]:
        update_parameters({'omega_c': omega_c[i], 'gating_type': None, 'architecture': 'LSTM', 'training_method': 'SL',\
            'n_train_batches': 6000, 'include_rule_signal': True, 'num_rule_tuned': 20, 'gate_pct': 0.5})
        save_fn = 'LSTM_SL_no_gating_rule_signal_omega' + str(i) + '_v' + str(j) + '.pkl'
        try_model(save_fn)



# LSTM - Supervised Learning - No gating
for j in range(1,5):
    for i in [5,7]:
        update_parameters({'omega_c': omega_c[i], 'gating_type': 'partial', 'architecture': 'LSTM', 'training_method': 'SL',\
            'n_train_batches': 6000, 'gate_pct': 0.5})
        save_fn = 'LSTM_SL_partial_gating_omega' + str(i) + '_v' + str(j) + '.pkl'
        try_model(save_fn)
