### Authors: Nicolas Y. Masse, Gregory D. Grant

import numpy as np
import tensorflow as tf
from itertools import product
import matplotlib.pyplot as plt

print("\n--> Loading parameters...")

##############################
### Independent parameters ###
##############################

global par

par = {
    # General parameters
    'save_dir'              : './savedir/',
    'loss_function'         : 'MSE',     # cross_entropy or MSE
    'stabilization'         : 'pathint', # 'EWC' (Kirkpatrick method) or 'pathint' (Zenke method)
    'learning_rate'         : 0.001,
    'save_analysis'         : False,
    'reset_weights'         : False,    # reset weights between tasks

    # Network configuration
    'synapse_config'        : None,     # Full is 'std_stf'
    'exc_inh_prop'          : 0.8,      # Literature 0.8, for EI off 1
    'var_delay'             : False,

    # Network shape
    'num_motion_tuned'      : 36,
    'num_fix_tuned'         : 20,
    'num_rule_tuned'        : 0,
    'n_hidden'              : 100,
    'n_dendrites'           : 1,

    # Euclidean shape
    'num_sublayers'         : 1,
    'neuron_dx'             : 1.0,
    'neuron_dy'             : 1.0,
    'neuron_dz'             : 10.0,

    # Timings and rates
    'dt'                    : 50,
    'learning_rate'         : 5e-3,
    'membrane_time_constant': 50,
    'connection_prob'       : 1.0,         # Usually 1

    # Variance values
    'clip_max_grad_val'     : 0.5,
    'input_mean'            : 0.0,
    'noise_in_sd'           : 0.1,
    'noise_rnn_sd'          : 0.5,

    # Task specs
    'task'                  : 'multistim',
    'n_tasks'               : 20,
    'multistim_trial_length': 2000,
    'mask_duration'         : 200,
    'dead_time'             : 200,

    # Tuning function data
    'num_motion_dirs'       : 8,
    'tuning_height'         : 4.0,        # magnitude scaling factor for von Mises
    'kappa'                 : 2.0,        # concentration scaling factor for von Mises

    # Cost parameters
    'spike_cost'            : 1e-7,
    'wiring_cost'           : 0, #1e-6,

    # Synaptic plasticity specs
    'tau_fast'              : 100,
    'tau_slow'              : 1500,
    'U_stf'                 : 0.15,
    'U_std'                 : 0.45,

    # Training specs
    'batch_size'            : 131,
    'n_train_batches'       : 5000,

    # Omega parameters
    'omega_c'               : 0.1,
    'omega_xi'              : 0.01,
    'EWC_fisher_num_batches': 32,   # number of batches when calculating EWC

    # Gating parameters
    'gating_type'           : 'XdG', # 'XdG', 'partial', 'split', None
    'gate_pct'              : 0.5,  # Num. gated hidden units for 'XdG' only
    'n_subnetworks'         : 4,    # Num. subnetworks for 'split' only

    # Save paths
    'save_fn'               : 'model_results.pkl',
    'ckpt_save_fn'          : 'model.ckpt',
    'ckpt_load_fn'          : 'model.ckpt',

}

############################
### Dependent parameters ###
############################


def update_parameters(updates):
    """
    Takes a list of strings and values for updating parameters in the parameter dictionary
    Example: updates = [(key, val), (key, val)]
    """
    for (key, val) in updates.items():
        par[key] = val
        print('Updating : ', key, ' -> ', val)
    update_dependencies()


def gen_gating():
    """
    Generate the gating signal to applied to all hidden units
    """
    par['gating'] = []

    for t in range(par['n_tasks']):
        gating_task = []
        gating_layer = np.zeros(par['n_hidden'], dtype=np.float32)
        for i in range(par['n_hidden']):

            if par['gating_type'] == 'XdG':
                if np.random.rand() < 1-par['gate_pct']:
                    gating_layer[i] = 1

            elif par['gating_type'] == 'split':
                if t%par['n_subnetworks'] == i%par['n_subnetworks']:
                    if np.random.rand() < 1-par['gate_pct']:
                        gating_layer[i] = 0.5
                    else:
                        gating_layer[i] = 1

            elif par['gating_type'] == 'partial':
                if np.random.rand() < 1-par['gate_pct']:
                    gating_layer[i] = 0.5
                else:
                    gating_layer[i] = 1

            elif par['gating_type'] is None:
                gating_layer[i] = 1

            gating_task.append(gating_layer)

        par['gating'].append(gating_task)

    par['gating'] = np.array(par['gating'])[:,:,0]


def initialize_weight(dims, connection_prob):
    w = np.random.gamma(shape=0.25, scale=1.0, size=dims)
    #w = np.random.uniform(low=0, high=0.5, size=dims)
    w *= (np.random.rand(*dims) < connection_prob)
    return np.float32(w)


def spectral_radius(A):
    if A.ndim == 3:
        return np.max(abs(np.linalg.eigvals(np.sum(A, axis=1))))
    else:
        return np.max(abs(np.linalg.eigvals(np.sum(A))))


def square_locs(num_locs, d1, d2):

    locs_per_side = np.int32(np.sqrt(num_locs))
    while locs_per_side**2 < num_locs:
        locs_per_side += 1

    x_set = np.repeat(d1*np.arange(locs_per_side)[:,np.newaxis], locs_per_side, axis=1).flatten()
    y_set = np.repeat(d2*np.arange(locs_per_side)[np.newaxis,:], locs_per_side, axis=0).flatten()
    locs  = np.stack([x_set, y_set])[:,:num_locs]

    locs[0,:] -= np.max(locs[0,:])/2
    locs[1,:] -= np.max(locs[1,:])/2

    return locs


def update_dependencies():
    """
    Updates all parameter dependencies
    """

    # Number of output neurons
    par['n_output'] = par['num_motion_dirs'] + 1

    # Number of input neurons
    par['n_input'] = par['num_motion_tuned'] + par['num_fix_tuned'] + par['num_rule_tuned']

    # General network shape
    par['shape'] = (par['n_input'], par['n_hidden'], par['n_output'])

    # Set up gating vectors for hidden layer
    gen_gating()

    # If num_inh_units is set > 0, then neurons can be either excitatory or
    # inihibitory; is num_inh_units = 0, then the weights projecting from
    # a single neuron can be a mixture of excitatory or inhibitory
    if par['exc_inh_prop'] < 1:
        par['EI'] = True
    else:
        par['EI']  = False

    par['num_exc_units'] = int(np.round(par['n_hidden']*par['exc_inh_prop']))
    par['num_inh_units'] = par['n_hidden'] - par['num_exc_units']
    par['EI_list'] = np.ones(par['n_hidden'], dtype=np.float32)
    par['EI_list'][::par['n_hidden']//par['num_inh_units']] = -1.

    par['EI_matrix'] = np.diag(par['EI_list'])

    # Membrane time constant of RNN neurons
    par['alpha_neuron'] = np.float32(par['dt'])/par['membrane_time_constant']
    # The standard deviation of the Gaussian noise added to each RNN neuron
    # at each time step
    par['noise_rnn'] = np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd']
    par['noise_in'] = np.sqrt(2/par['alpha_neuron'])*par['noise_in_sd'] # since term will be multiplied by par['alpha_neuron']

    # Set trial step length
    par['num_time_steps'] = par['multistim_trial_length']//par['dt']


    ####################################################################
    ### Setting up assorted intial weights, biases, and other values ###
    ####################################################################

    par['h_init'] = 0.1*np.ones((par['n_hidden'], par['batch_size']), dtype=np.float32)

    par['input_to_hidden_dims'] = [par['n_hidden'], par['n_dendrites'], par['n_input']]
    par['hidden_to_hidden_dims'] = [par['n_hidden'], par['n_dendrites'], par['n_hidden']]
    par['hidden_to_output_dims'] = [par['n_output'], par['n_hidden']]

    # Initialize input weights
    par['w_in0'] = initialize_weight(par['input_to_hidden_dims'], par['connection_prob'])
    par['w_in_mask'] = np.ones(par['input_to_hidden_dims'], dtype = np.float32)

    # Initialize starting recurrent weights
    # If excitatory/inhibitory neurons desired, initializes with random matrix with
    #   zeroes on the diagonal
    # If not, initializes with a diagonal matrix
    if par['EI']:
        par['w_rnn0'] = initialize_weight(par['hidden_to_hidden_dims'], par['connection_prob'])

        for i in range(par['n_hidden']):
            par['w_rnn0'][i,:,i] = 0
        par['w_rnn_mask'] = np.ones((par['hidden_to_hidden_dims']), dtype=np.float32) - np.eye(par['n_hidden'])[:,np.newaxis,:]
        #par['w_rnn0'][:,:,par['num_exc_units']:] *= par['exc_inh_prop']/(1-par['exc_inh_prop'])
    else:
        par['w_rnn0'] = np.concatenate([np.float32(0.5*np.eye(par['n_hidden']))[:,np.newaxis,:]]*par['n_dendrites'], axis=1)
        par['w_rnn_mask'] = np.ones((par['hidden_to_hidden_dims']), dtype=np.float32)

    par['b_rnn0'] = np.zeros((par['n_hidden'], 1), dtype=np.float32)



    # Effective synaptic weights are stronger when no short-term synaptic plasticity
    # is used, so the strength of the recurrent weights is reduced to compensate
    if par['synapse_config'] == None:
        par['w_rnn0'] = par['w_rnn0']/(spectral_radius(par['w_rnn0']))


    # Initialize output weights and biases
    par['w_out0'] = initialize_weight(par['hidden_to_output_dims'], par['connection_prob'])
    par['b_out0'] = np.zeros((par['n_output'], 1), dtype=np.float32)
    par['w_out_mask'] = np.ones(par['hidden_to_output_dims'], dtype=np.float32)

    if par['EI']:
        par['ind_inh'] = np.where(par['EI_list'] == -1)[0]
        par['w_out0'][:, par['ind_inh']] = 0
        par['w_out_mask'][:, par['ind_inh']] = 0

    # Defining sublayers for the hidden layer
    n_per_sub = par['n_hidden']//par['num_sublayers']
    sublayers = []
    sublayer_sizes = []
    for i in range(par['num_sublayers']):
        if i == par['num_sublayers'] - 1:
            app = par['n_hidden']%par['num_sublayers']
        else:
            app = 0
        sublayers.append(range(i*n_per_sub,(i+1)*n_per_sub+app))
        sublayer_sizes.append(n_per_sub+app)

    # Determine physical sublayer positions
    input_pos = np.zeros([par['n_input'], 3])
    hidden_pos = np.zeros([par['n_hidden'], 3])
    output_pos = np.zeros([par['n_output'], 3])

    # Build layer geometry
    input_pos[:,0:2] = square_locs(par['n_input'], par['neuron_dx'], par['neuron_dy']).T
    input_pos[:,2] = 0

    for i, (s, l) in enumerate(zip(sublayers, sublayer_sizes)):
        hidden_pos[s,0:2] = square_locs(l, par['neuron_dx'], par['neuron_dy']).T
        hidden_pos[s,2] = (i+1)*par['neuron_dz']

    output_pos[:,0:2] = square_locs(par['n_output'], par['neuron_dx'], par['neuron_dy']).T
    output_pos[:,2] = np.max(hidden_pos[:,2]) + par['neuron_dz']

    # Apply physical positions to relative positional matrix
    par['w_in_pos'] = np.zeros(par['input_to_hidden_dims'])
    for i,j in product(range(par['n_input']), range(par['n_hidden'])):
        par['w_in_pos'][j,:,i] = np.sqrt(np.sum(np.square(input_pos[i,:] - hidden_pos[j,:])))

    par['w_rnn_pos'] = np.zeros(par['hidden_to_hidden_dims'])
    for i,j in product(range(par['n_hidden']), range(par['n_hidden'])):
        par['w_rnn_pos'][j,:,i] = np.sqrt(np.sum(np.square(hidden_pos[i,:] - hidden_pos[j,:])))

    par['w_out_pos'] = np.zeros(par['hidden_to_output_dims'])
    for i,j in product(range(par['n_hidden']), range(par['n_output'])):
        par['w_out_pos'][j,i] = np.sqrt(np.sum(np.square(hidden_pos[i,:] - output_pos[j,:])))

    # Specify connections to sublayers
    for i in range(1, par['num_sublayers']):
        par['w_in0'][sublayers[i],:,:] = 0
        par['w_in_mask'][sublayers[i],:,:] = 0

    # Only allow connections between adjacent sublayers
    for i,j in product(range(par['num_sublayers']), range(par['num_sublayers'])):
        if np.abs(i-j) > 1:
            for k,m in product(sublayers[i], sublayers[j]):
                par['w_rnn0'][k,:,m] = 0
                par['w_rnn_mask'][k,:,m] = 0

    # Specify connections from sublayers
    for i in range(par['num_sublayers'] - 1):
        par['w_out0'][:, sublayers[i]] = 0
        par['w_out_mask'][:, sublayers[i]] = 0


    """
    Setting up synaptic parameters
    0 = static
    1 = facilitating
    2 = depressing
    """
    par['synapse_type'] = np.zeros(par['n_hidden'], dtype=np.int8)

    # only facilitating synapses
    if par['synapse_config'] == 'stf':
        par['synapse_type'] = np.ones(par['n_hidden'], dtype=np.int8)

    # only depressing synapses
    elif par['synapse_config'] == 'std':
        par['synapse_type'] = 2*np.ones(par['n_hidden'], dtype=np.int8)

    # even numbers facilitating, odd numbers depressing
    elif par['synapse_config'] == 'std_stf':
        par['synapse_type'] = np.ones(par['n_hidden'], dtype=np.int8)
        par['ind'] = range(1,par['n_hidden'],2)
        par['synapse_type'][par['ind']] = 2

    par['alpha_stf'] = np.ones((par['n_hidden'], 1), dtype=np.float32)
    par['alpha_std'] = np.ones((par['n_hidden'], 1), dtype=np.float32)
    par['U'] = np.ones((par['n_hidden'], 1), dtype=np.float32)

    # initial synaptic values
    par['syn_x_init'] = np.zeros((par['n_hidden'], par['batch_size']), dtype=np.float32)
    par['syn_u_init'] = np.zeros((par['n_hidden'], par['batch_size']), dtype=np.float32)

    for i in range(par['n_hidden']):
        if par['synapse_type'][i] == 1:
            par['alpha_stf'][i,0] = par['dt']/par['tau_slow']
            par['alpha_std'][i,0] = par['dt']/par['tau_fast']
            par['U'][i,0] = 0.15
            par['syn_x_init'][i,:] = 1
            par['syn_u_init'][i,:] = par['U'][i,0]

        elif par['synapse_type'][i] == 2:
            par['alpha_stf'][i,0] = par['dt']/par['tau_fast']
            par['alpha_std'][i,0] = par['dt']/par['tau_slow']
            par['U'][i,0] = 0.45
            par['syn_x_init'][i,:] = 1
            par['syn_u_init'][i,:] = par['U'][i,0]


update_dependencies()
print("--> Parameters successfully loaded.\n")
