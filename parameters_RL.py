### Authors: Nicolas Y. Masse, Gregory D. Grant

import numpy as np
import tensorflow as tf
from itertools import product, chain
import matplotlib.pyplot as plt

print("\n--> Loading parameters...")

##############################
### Independent parameters ###
##############################

global par

par = {
    # General parameters
    'save_dir'              : './savedir/',
    'stabilization'         : 'pathint', # 'EWC' (Kirkpatrick method) or 'pathint' (Zenke method)
    'save_analysis'         : False,
    'reset_weights'         : False,    # reset weights between tasks

    # Network configuration
    'synapse_config'        : 'std_stf',     # Full is 'std_stf'
    'exc_inh_prop'          : 0.8,      # Literature 0.8, for EI off 1
    'var_delay'             : False,
    'LSTM'                  : True,

    # Network shape
    'num_motion_tuned'      : 64,
    'num_fix_tuned'         : 4,
    'num_rule_tuned'        : 0,
    'n_hidden'              : 256,
    'n_d_hidden'            : 100, # distill hidden neurons
    'n_val_hidden'          : 200,
    'n_dendrites'           : 1, # don't use for now
    'n_val'                 : 1,
    'include_rule_signal'   : False,

    # Euclidean shape
    'num_sublayers'         : 1,
    'neuron_dx'             : 1.0,
    'neuron_dy'             : 1.0,
    'neuron_dz'             : 10.0,

    # Timings and rates
    'dt'                    : 20,
    'learning_rate'         : 5e-4,
    'membrane_time_constant': 100,
    'connection_prob'       : 1.0,
    'discount_rate'         : 0.,

    # Variance values
    'clip_max_grad_val'     : 1.0,
    'input_mean'            : 0.0,
    'noise_in_sd'           : 0.0,
    'noise_rnn_sd'          : 0.05,

    # Task specs
    'task'                  : 'multistim',
    'n_tasks'               : 20,
    'multistim_trial_length': 2000,
    'mask_duration'         : 0,
    'dead_time'             : 200,

    # Tuning function data
    'num_motion_dirs'       : 8,
    'tuning_height'         : 4.0,        # magnitude scaling factor for von Mises
    'kappa'                 : 2.0,        # concentration scaling factor for von Mises

    # Cost parameters
    'spike_cost'            : 1e-7,
    'weight_cost'           : 0.,
    'entropy_cost'          : 0.0,
    'drop_rate'             : 0.0,
    'val_cost'              : 0.01,

    # Synaptic plasticity specs
    'tau_fast'              : 100,
    'tau_slow'              : 1000,
    'U_stf'                 : 0.15,
    'U_std'                 : 0.45,

    # Training specs
    'batch_size'            : 256,
    'n_train_batches'       : 50000,

    # Omega parameters
    'omega_c'               : 0.1,
    'omega_xi'              : 0.01,
    'EWC_fisher_num_batches': 16,   # number of batches when calculating EWC

    # Gating parameters
    'gating_type'           : 'XdG', # 'XdG', 'partial', 'split', None
    'gate_pct'              : 0.8,  # Num. gated hidden units for 'XdG' only
    'n_subnetworks'         : 4,    # Num. subnetworks for 'split' only

    # Stimulus parameters
    'fix_break_penalty'     : -1.,
    'wrong_choice_penalty'  : -0.01,
    'correct_choice_reward' : 1.,

    # Save paths
    'save_fn'               : 'model_results.pkl',
    'ckpt_save_fn'          : 'model.ckpt',
    'ckpt_load_fn'          : 'model.ckpt',

    'constrain_input_weights': False

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
    par['val_gating'] = []

    for t in range(par['n_tasks']):
        gating_task = np.zeros(par['n_hidden'], dtype=np.float32)
        val_gating_task = np.zeros(par['n_val_hidden'], dtype=np.float32)
        for i in range(par['n_val_hidden']):
            if par['gating_type'] == 'XdG':
                if np.random.rand() < 1-par['gate_pct']:
                    val_gating_task[i] = 1
                elif par['gating_type'] is None:
                    val_gating_task[i] = 1

        for i in range(par['n_hidden']):

            if par['gating_type'] == 'XdG':
                if np.random.rand() < 1-par['gate_pct']:
                    gating_task[i] = 1

            elif par['gating_type'] == 'split':
                if t%par['n_subnetworks'] == i%par['n_subnetworks']:
                    if np.random.rand() < 1-par['gate_pct']:
                        gating_task[i] = 0.5
                    else:
                        gating_task[i] = 1

            elif par['gating_type'] == 'partial':
                if np.random.rand() < 1-par['gate_pct']:
                    gating_task[i] = 0.5
                else:
                    gating_task[i] = 1

            elif par['gating_type'] is None:
                gating_task[i] = 1

        par['gating'].append(gating_task)
        par['val_gating'].append(val_gating_task)



def initialize_weight(dims, connection_prob):
    w = np.random.gamma(shape=0.25, scale=1.0, size=dims)
    #w = np.random.uniform(low=0, high=0.5, size=dims)
    w *= (np.random.rand(*dims) < connection_prob)
    return np.float32(w)


def spectral_radius(A):
    if A.ndim == 3:
        return np.max(abs(np.linalg.eigvals(np.sum(A, axis=1))))
    else:
        return np.max(abs(np.linalg.eigvals(A)))


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

    if par['exc_inh_prop'] < 1 and not par['LSTM']:
        par['EI'] = True
    elif par['LSTM']:
        print('Using LSTM networks; setting to EI to False')
        par['EI'] = False
        par['exc_inh_prop'] = 1.
        par['synapse_config'] = False
        par['spike_cost'] = 0.

    # Number of output neurons
    par['n_output'] = par['num_motion_dirs'] + 1
    par['n_pol'] = par['num_motion_dirs'] + 1

    # Number of input neurons
    par['n_input'] = par['num_motion_tuned'] + par['num_fix_tuned'] + par['num_rule_tuned']

    # General network shape
    par['shape'] = (par['n_input'], par['n_hidden'], par['n_output'])

    par['dt_sec'] = par['dt']/1000
    # Set up gating vectors for hidden layer
    gen_gating()

    # If num_inh_units is set > 0, then neurons can be either excitatory or
    # inihibitory; is num_inh_units = 0, then the weights projecting from
    # a single neuron can be a mixture of excitatory or inhibitory
    par['EI'] = True if par['exc_inh_prop'] < 1 else False


    par['num_exc_units'] = int(np.round(par['n_hidden']*par['exc_inh_prop']))
    par['num_inh_units'] = par['n_hidden'] - par['num_exc_units']
    par['EI_list'] = np.ones(par['n_hidden'], dtype=np.float32)
    if par['EI']:
        n = par['n_hidden']//par['num_inh_units']
        par['ind_inh'] = np.arange(n-1,par['n_hidden'],n)
        par['EI_list'][par['ind_inh']] = -1.
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

    par['h_init'] = 0.1*np.ones((par['batch_size'], par['n_hidden']), dtype=np.float32)
    par['h_d_init'] = 0.1*np.ones((par['batch_size'], par['n_d_hidden']), dtype=np.float32)

    # Initialize input weights
    c = 0.1
    if par['EI']:
        par['W_rnn_init'] = c*np.float32(np.random.gamma(shape=0.25, scale=1.0, size = [par['n_hidden'], par['n_hidden']]))
        par['W_rnn_mask'] = np.ones((par['n_hidden'], par['n_hidden']), dtype=np.float32) - np.eye(par['n_hidden'])
        par['W_rnn_init'] *= par['W_rnn_mask']

        par['W_d_rnn_init'] = 0.05*np.float32(np.random.gamma(shape=0.25, scale=1.0, size = [par['n_d_hidden'], par['n_d_hidden']]))
        par['W_d_rnn_mask'] = np.ones((par['n_d_hidden'], par['n_d_hidden']), dtype=np.float32) - np.eye(par['n_d_hidden'])
        par['W_d_rnn_init'] *= par['W_d_rnn_mask']
    else:
        par['W_rnn_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_hidden']]))
        par['W_rnn_mask'] = np.ones((par['n_hidden'], par['n_hidden']), dtype=np.float32)
    """
    if par['synapse_config'] == None:
        par['W_rnn_init'] /= (spectral_radius(par['W_rnn_init'])/2)
    """
    #par['W_rnn_init'][par['ind_inh'],: ] *= 4

    #s = np.dot(par['EI_matrix'], par['W_rnn_init'])
    #plt.imshow(s, aspect = 'auto')
    #plt.colorbar()
    #plt.show()

    par['reset_weight_mask'] = np.ones((par['n_input'], par['n_hidden']), dtype=np.float32)
    par['reset_weight_mask'][-par['num_rule_tuned']:,:] = 0.

    #par['W_out_init'] = np.float32(np.random.gamma(shape=0.25, scale=1.0, size = [par['n_hidden'], par['n_output']]))
    par['W_out_init'] = np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_output']]))

    #par['W_in_init'] = np.float32(np.random.gamma(shape=0.25, scale=1.0, size = [par['n_input'], par['n_hidden']]))
    #par['W_in_init'] = np.float32(np.random.uniform(-0.25, 0.25, size = [par['n_input'], par['n_hidden']]))
    par['W_in_init'] = c*np.float32(np.random.gamma(shape=0.25, scale=1.0, size = [par['n_input'], par['n_hidden']]))
    #par['W_d_in_init'] = np.float32(np.random.uniform(-c, c, size = [par['n_input'], par['n_d_hidden']]))
    #par['W_in_init'][-par['num_rule_tuned']:, :] -= 0.5*np.float32(np.random.gamma(shape=0.25, scale=1.0, size = [par['num_rule_tuned'], par['n_hidden']]))

    par['b_rnn_init'] = np.zeros((1,par['n_hidden']), dtype = np.float32)
    par['b_d_rnn_init'] = np.zeros((1,par['n_d_hidden']), dtype = np.float32)
    par['b_out_init'] = np.zeros((1,par['n_output']), dtype = np.float32)

    par['W_out_mask'] = np.ones((par['n_hidden'], par['n_output']), dtype=np.float32)
    par['W_in_mask'] = np.ones((par['n_input'], par['n_hidden']), dtype=np.float32)

    if par['EI']:
        par['W_out_init'][par['ind_inh'], :] = 0
        par['W_out_mask'][par['ind_inh'], :] = 0

    # RL
    par['W_pol_out_init'] = np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_pol']]))
    par['b_pol_out_init'] = np.zeros((1,par['n_pol']), dtype = np.float32)
    par['W_d_out_init'] = np.float32(np.random.uniform(-c, c, size = [par['n_d_hidden'], par['n_pol']]))
    par['b_d_out_init'] = np.float32(np.random.uniform(-c, c, size = [1, par['n_pol']]))


    par['W_val_out_init'] = np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_val']]))
    par['b_val_out_init'] = np.zeros((1,par['n_val']), dtype = np.float32)


    if par['LSTM']:
        c = 0.05
        par['Wf_init'] =  c*np.float32(np.random.uniform(-c, c, size = [par['n_input'], par['n_hidden']]))
        par['Wi_init'] =  c*np.float32(np.random.uniform(-c, c, size = [par['n_input'], par['n_hidden']]))
        par['Wo_init'] =  c*np.float32(np.random.uniform(-c, c, size = [par['n_input'], par['n_hidden']]))
        par['Wc_init'] =  c*np.float32(np.random.uniform(-c, c, size = [par['n_input'], par['n_hidden']]))

        par['Uf_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_hidden']]))
        par['Ui_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_hidden']]))
        par['Uo_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_hidden']]))
        par['Uc_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_hidden']]))


        par['bf_init'] = np.zeros((1, par['n_hidden']), dtype = np.float32)
        par['bi_init'] = np.zeros((1, par['n_hidden']), dtype = np.float32)
        par['bo_init'] = np.zeros((1, par['n_hidden']), dtype = np.float32)
        par['bc_init'] = np.zeros((1, par['n_hidden']), dtype = np.float32)

    """
    par['W_val_out0_init'] = np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_val_hidden']]))
    #par['W_act_val_out0_init'] = np.float32(np.random.uniform(-c, c, size = [par['n_pol'], par['n_val_hidden']]))
    par['b_val_out0_init'] = np.zeros((1, par['n_val_hidden']), dtype = np.float32)
    par['W_val_out1_init'] = np.float32(np.random.uniform(-c, c, size = [par['n_val_hidden'], par['n_val_hidden']]))
    par['b_val_out1_init'] = np.zeros((1, par['n_val_hidden']), dtype = np.float32)

    par['W_val_out2_init'] = np.float32(np.random.uniform(-c, c, size = [par['n_val_hidden'], par['n_val']]))
    par['b_val_out2_init'] = np.zeros((1,par['n_val']), dtype = np.float32)
    """

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
    """
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
    """



    # Specify connections to sublayers
    for i in range(1, par['num_sublayers']):
        par['W_in_init'][:par['num_motion_tuned']+par['num_fix_tuned'], sublayers[i]] = 0
        par['W_in_mask'][:par['num_motion_tuned']+par['num_fix_tuned'], sublayers[i]] = 0

    if par['constrain_input_weights']:
        k = 0
        # motion tuned only projects to sublayers[i][0:-1:3]
        #for i in chain(sublayers[k][1:-1:3], sublayers[k][2:-1:3]):
        for i in chain(sublayers[k][0:-1:6], sublayers[k][1:-1:6], sublayers[k][2:-1:6]):
            par['W_in_init'][:par['num_motion_tuned']//2, i] = 0
            par['W_in_mask'][:par['num_motion_tuned']//2, i] = 0
        for i in chain(sublayers[k][0:-1:6], sublayers[k][4:-1:6], sublayers[k][5:-1:6]):
            par['W_in_init'][par['num_motion_tuned']//2:par['num_motion_tuned'], i] = 0
            par['W_in_mask'][par['num_motion_tuned']//2:par['num_motion_tuned'], i] = 0

        # fixation tuned only prohject to sublayers[i][0:-1:3]
        for i in chain(sublayers[k][1:-1:6], sublayers[k][2:-1:6], sublayers[k][3:-1:6],\
            sublayers[k][4:-1:6], sublayers[k][5:-1:6]):
            par['W_in_init'][par['num_motion_tuned']:par['num_motion_tuned']+par['num_fix_tuned'], i] = 0
            par['W_in_mask'][par['num_motion_tuned']:par['num_motion_tuned']+par['num_fix_tuned'], i] = 0






    # Only allow connections between adjacent sublayers
    for i,j in product(range(par['num_sublayers']), range(par['num_sublayers'])):
        if np.abs(i-j) > 1:
            for k,m in product(sublayers[i], sublayers[j]):
                par['W_rnn_init'][m,k] = 0
                par['W_rnn_mask'][m,k] = 0

    # Specify connections from sublayers
    for i in range(par['num_sublayers'] - 1):
        par['W_out_init'][sublayers[i], :] = 0
        par['W_out_mask'][sublayers[i], :] = 0

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
        par['synapse_type'] = 2*np.ones(par['n_hidden'], dtype=np.int8)
        ind = range(1,par['n_hidden'],2)
        #par['synapse_type'][par['ind_inh']] = 1
        par['synapse_type'][ind] = 1

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

    par['alpha_stf'] = np.transpose(par['alpha_stf'])
    par['alpha_std'] = np.transpose(par['alpha_std'])
    par['U'] = np.transpose(par['U'])
    par['syn_x_init'] = np.transpose(par['syn_x_init'])
    par['syn_u_init'] = np.transpose(par['syn_u_init'])


update_dependencies()
print("--> Parameters successfully loaded.\n")
