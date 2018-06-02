### Authors: Nicolas Y. Masse, Gregory D. Grant

import tensorflow as tf
import numpy as np
import stimulus
import AdamOpt
from parameters import *
import os, time
import pickle
import convolutional_layers
from itertools import product
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi

# Ignore startup TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

###################
### Model setup ###
###################
class Model:

    def __init__(self, input_data, target_data, gating, mask):

        # Load the input activity, the target data, and the training mask
        # for this batch of trials
        self.input_data         = tf.unstack(input_data, axis=0)
        self.gating             = tf.reshape(gating, [1,-1])
        self.target_data        = tf.unstack(target_data, axis=0)
        self.mask               = tf.unstack(mask, axis=0)
        self.W_ei               = tf.constant(par['EI_matrix'])

        # Build the TensorFlow graph
        self.hidden_state_hist = []
        self.syn_x_hist = []
        self.syn_u_hist = []
        self.output = []
        self.run_model()

        # Train the model
        self.optimize()


    def run_model(self):

        with tf.variable_scope('rnn'):
            W_in  = tf.get_variable('W_in', initializer=par['W_in_init'], trainable=True)
            W_rnn = tf.get_variable('W_rnn', initializer=par['W_rnn_init'], trainable=True)
            b_rnn = tf.get_variable('b_rnn', initializer=par['b_rnn_init'], trainable=True)

        with tf.variable_scope('output'):
            W_out = tf.get_variable('W_out', initializer=par['W_out_init'], trainable=True)
            b_out = tf.get_variable('b_out', initializer=par['b_out_init'], trainable=True)

        if par['EI']:
            W_rnn = tf.matmul(self.W_ei, tf.nn.relu(W_rnn))


        h = tf.constant(par['h_init'])
        syn_x = tf.constant(par['syn_x_init'])
        syn_u = tf.constant(par['syn_u_init'])
        for x in self.input_data:

            if par['synapse_config'] == 'std_stf':
                # implement both synaptic short term facilitation and depression
                syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_u*syn_x*h
                syn_u += par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h
                syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
                syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
                h_post = syn_u*syn_x*h

            elif par['synapse_config'] == 'std':
                # implement synaptic short term derpression, but no facilitation
                # we assume that syn_u remains constant at 1
                syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_x*h
                syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
                syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
                h_post = syn_x*h

            elif par['synapse_config'] == 'stf':
                # implement synaptic short term facilitation, but no depression
                # we assume that syn_x remains constant at 1
                syn_u += par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h
                syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
                h_post = syn_u*h

            else:
                # no synaptic plasticity
                h_post = h



            # Hidden State
            h = self.gating*tf.nn.relu((1-par['alpha_neuron'])*h + par['alpha_neuron']*(tf.matmul(x, W_in) + \
                tf.matmul(h_post, W_rnn) + b_rnn) + tf.random_normal(h.shape, 0, par['noise_rnn'], dtype=tf.float32))

            #h = tf.minimum(50., h)

            # Output State
            y = tf.matmul(h, W_out) + b_out

            # Bookkeeping lists
            self.hidden_state_hist.append(h)
            self.syn_x_hist.append(syn_x)
            self.syn_u_hist.append(syn_u)
            self.output.append(y)



    def optimize(self):

        # Use all trainable variables, except those in the convolutional layers
        self.variables = [var for var in tf.trainable_variables() if not var.op.name.find('conv')==0]
        adam_optimizer = AdamOpt.AdamOpt(self.variables, learning_rate = par['learning_rate'])

        previous_weights_mu_minus_1 = {}
        reset_prev_vars_ops = []
        self.big_omega_var = {}
        aux_losses = []

        for var in self.variables:
            self.big_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            previous_weights_mu_minus_1[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            aux_losses.append(par['omega_c']*tf.reduce_sum(tf.multiply(self.big_omega_var[var.op.name], \
               tf.square(previous_weights_mu_minus_1[var.op.name] - var) )))
            reset_prev_vars_ops.append( tf.assign(previous_weights_mu_minus_1[var.op.name], var ) )

        self.aux_loss = tf.add_n(aux_losses)

        self.spike_loss = par['spike_cost']*tf.reduce_mean(tf.square(self.hidden_state_hist))


        self.task_loss = tf.reduce_mean([mask*tf.nn.softmax_cross_entropy_with_logits(logits = y, \
            labels = target, dim=1) for y, target, mask in zip(self.output, self.target_data, self.mask)])


        output_softmax = [tf.nn.softmax(y, dim = 1) for y in self.output]
        self.entropy_loss = -par['entropy_cost']*tf.reduce_mean([m*tf.reduce_sum(out_sm*tf.log(1e-7+out_sm), axis = 1) \
            for (out_sm,m) in zip(output_softmax, self.mask)])

        """
        with tf.variable_scope('rnn', reuse = True):
            W_in  = tf.get_variable('W_in')
            W_rnn = tf.get_variable('W_rnn')

        active_weights_rnn = tf.matmul(tf.reshape(self.gating,[-1,1]), tf.reshape(self.gating,[1,-1]))
        active_weights_in = tf.tile(tf.reshape(self.gating,[1,-1]),[par['n_input'], 1])
        self.weight_loss = par['weight_cost']*(tf.reduce_mean(active_weights_in*W_in**2) + tf.reduce_mean(tf.nn.relu(active_weights_rnn*W_rnn)**2))
        """
        # Gradient of the loss+aux function, in order to both perform training and to compute delta_weights
        with tf.control_dependencies([self.task_loss, self.aux_loss, self.spike_loss, self.entropy_loss ]):
            self.train_op = adam_optimizer.compute_gradients(self.task_loss + self.aux_loss + self.spike_loss - self.entropy_loss)

        # Stabilizing weights
        if par['stabilization'] == 'pathint':
            # Zenke method
            self.pathint_stabilization(adam_optimizer, previous_weights_mu_minus_1)

        elif par['stabilization'] == 'EWC':
            # Kirkpatrick method
            self.EWC()

        self.reset_prev_vars = tf.group(*reset_prev_vars_ops)
        self.reset_adam_op = adam_optimizer.reset_params()

        self.reset_weights()

        self.make_recurrent_weights_positive()


    def reset_weights(self):

        reset_weights = []

        for var in self.variables:
            if 'b' in var.op.name:
                # reset biases to 0
                reset_weights.append(tf.assign(var, var*0.))
            elif 'W' in var.op.name:
                # reset weights to initial-like conditions
                new_weight = initialize_weight(var.shape, par['connection_prob'])
                reset_weights.append(tf.assign(var,new_weight))

        self.reset_weights = tf.group(*reset_weights)

    def make_recurrent_weights_positive(self):

        reset_weights = []
        for var in self.variables:
            if 'W_rnn' in var.op.name:
                # make all negative weights slightly positive
                reset_weights.append(tf.assign(var,tf.maximum(1e-9, var)))

        self.reset_rnn_weights = tf.group(*reset_weights)


    def EWC(self):

        # Kirkpatrick method
        epsilon = 1e-5
        fisher_ops = []
        opt = tf.train.GradientDescentOptimizer(1)

        # model results p(y|x, theta)
        p_theta = tf.nn.softmax(self.output, dim = 1)
        # sample label from p(y|x, theta)
        class_ind = tf.multinomial(p_theta, 1)
        class_ind_one_hot = tf.reshape(tf.one_hot(class_ind, par['layer_dims'][-1]), \
            [par['batch_size'], par['layer_dims'][-1]])
        # calculate the gradient of log p(y|x, theta)
        log_p_theta = tf.unstack(class_ind_one_hot*tf.log(p_theta + epsilon), axis = 0)
        for lp in log_p_theta:
            grads_and_vars = opt.compute_gradients(lp)
            for grad, var in grads_and_vars:
                fisher_ops.append(tf.assign_add(self.big_omega_var[var.op.name], \
                    grad*grad/par['batch_size']/par['EWC_fisher_num_batches']))

        self.update_big_omega = tf.group(*fisher_ops)


    def pathint_stabilization(self, adam_optimizer, previous_weights_mu_minus_1):
        # Zenke method

        optimizer_task = tf.train.GradientDescentOptimizer(learning_rate =  1.0)
        small_omega_var = {}

        reset_small_omega_ops = []
        update_small_omega_ops = []
        update_big_omega_ops = []
        initialize_prev_weights_ops = []

        for var in self.variables:

            small_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            reset_small_omega_ops.append( tf.assign( small_omega_var[var.op.name], small_omega_var[var.op.name]*0.0 ) )
            update_big_omega_ops.append( tf.assign_add( self.big_omega_var[var.op.name], tf.div(tf.nn.relu(small_omega_var[var.op.name]), \
            	(par['omega_xi'] + tf.square(var-previous_weights_mu_minus_1[var.op.name])))))


        # After each task is complete, call update_big_omega and reset_small_omega
        self.update_big_omega = tf.group(*update_big_omega_ops)

        # Reset_small_omega also makes a backup of the final weights, used as hook in the auxiliary loss
        self.reset_small_omega = tf.group(*reset_small_omega_ops)

        # This is called every batch
        with tf.control_dependencies([self.train_op]):
            self.delta_grads = adam_optimizer.return_delta_grads()
            self.gradients = optimizer_task.compute_gradients(self.task_loss)
            for grad,var in self.gradients:
                update_small_omega_ops.append(tf.assign_add(small_omega_var[var.op.name], -self.delta_grads[var.op.name]*grad ) )
            self.update_small_omega = tf.group(*update_small_omega_ops) # 1) update small_omega after each train!


def main(save_fn=None, gpu_id = None):

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # train the convolutional layers with the CIFAR-10 dataset
    # otherwise, it will load the convolutional weights from the saved file
    if (par['task'] == 'cifar' or par['task'] == 'imagenet') and par['train_convolutional_layers']:
        convolutional_layers.ConvolutionalLayers()

    print('\nRunning model.\n')

    # Reset TensorFlow graph
    tf.reset_default_graph()

    # Create placeholders for the model
    # input_data, target_data, gating, mask

    x  = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size'], par['n_input']], 'stim')
    target   = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size'], par['n_output']], 'out')
    mask   = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size']], 'mask')
    gating = tf.placeholder(tf.float32, [par['n_hidden']], 'gating')

    stim = stimulus.MultiStimulus()
    accuracy_full = []
    accuracy_grid = np.zeros((par['n_tasks'], par['n_tasks']))


    key_info = ['synapse_config','spike_cost','weight_cost','entropy_cost','omega_c','omega_xi',\
        'constrain_input_weights','num_sublayers','n_hidden','noise_rnn_sd','learning_rate','gating_type', 'gate_pct']
    print('Key info')
    for k in key_info:
        print(k, ' ', par[k])

    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            model = Model(x, target, gating, mask)

        sess.run(tf.global_variables_initializer())
        t_start = time.time()
        sess.run(model.reset_prev_vars)

        for task in range(0,par['n_tasks']):

            for i in range(par['n_train_batches']):

                # make batch of training data
                name, stim_in, y_hat, mk, _ = stim.generate_trial(task)

                if par['stabilization'] == 'pathint':
                    _, _, loss, AL, spike_loss, ent_loss, output = sess.run([model.train_op, \
                        model.update_small_omega, model.task_loss, model.aux_loss, model.spike_loss, \
                        model.entropy_loss, model.output], \
                        feed_dict = {x:stim_in, target:y_hat, gating:par['gating'][task], mask:mk})
                    sess.run([model.reset_rnn_weights])
                    if loss < 0.005 and AL < 0.0004 + 0.0002*task:
                        break

                elif par['stabilization'] == 'EWC':
                    _, loss, AL = sess.run([model.train_op, model.task_loss, model.aux_loss], feed_dict = \
                        {x:stim_in, target:y_hat, gating:par['gating'][task], mask:mk})

                if i%100 == 0:
                    acc = get_perf(y_hat, output, mk)
                    print('Iter ', i, 'Task name ', name, ' accuracy', acc, ' loss ', loss, ' aux loss', AL, ' spike loss', spike_loss, \
                        ' entropy loss', ent_loss)


            # Test all tasks at the end of each learning session
            num_reps = 10
            for (task_prime, r) in product(range(task+1), range(num_reps)):

                # make batch of training data
                name, stim_in, y_hat, mk, _ = stim.generate_trial(task_prime)

                output,_ = sess.run([model.output, model.syn_x_hist], feed_dict = {x:stim_in, gating:par['gating'][task_prime]})
                acc = get_perf(y_hat, output, mk)
                accuracy_grid[task,task_prime] += acc/num_reps

            print('Accuracy grid after task {}:'.format(task))
            print(accuracy_grid[task,:])
            print('')

            # Update big omegaes, and reset other values before starting new task
            if par['stabilization'] == 'pathint':
                big_omegas = sess.run([model.update_big_omega, model.big_omega_var])
            elif par['stabilization'] == 'EWC':
                for n in range(par['EWC_fisher_num_batches']):
                    name, stim_in, y_hat, mk, _ = stim.generate_trial(task)
                    big_omegas = sess.run([model.update_big_omega,model.big_omega_var], feed_dict = \
                        {x:stim_in, target:y_hat, gating:par['gating'][task], mask:mk})



            # Reset the Adam Optimizer, and set the previous parater values to their current values
            sess.run(model.reset_adam_op)
            sess.run(model.reset_prev_vars)
            if par['stabilization'] == 'pathint':
                sess.run(model.reset_small_omega)

            # reset weights between tasks if called upon
            if par['reset_weights']:
                sess.run(model.reset_weights)




        if par['save_analysis']:
            save_results = {'task': task, 'accuracy': accuracy, 'accuracy_full': accuracy_full, \
                            'accuracy_grid': accuracy_grid, 'big_omegas': big_omegas, 'par': par}
            pickle.dump(save_results, open(par['save_dir'] + save_fn, 'wb'))

    print('\nModel execution complete.')

def get_perf(target, output, mask):

    """
    Calculate task accuracy by comparing the actual network output to the desired output
    only examine time points when test stimulus is on
    in another words, when target[:,:,-1] is not 0
    """
    output = np.stack(output, axis=0)
    mk = mask*np.reshape(target[:,:,-1] == 0, (par['batch_size'], par['num_time_steps'], 1))

    target = np.argmax(target, axis = 2)
    output = np.argmax(output, axis = 2)

    return np.sum(np.float32(target == output)*np.squeeze(mk))/np.sum(mk)

#main('testing')
