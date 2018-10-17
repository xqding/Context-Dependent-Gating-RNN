# Author: NY Masse + GD Grant (modified by M Rosen)
import numpy as np
from parameters import par
import tensorflow as tf


class DynamicStimulus_tf:

    def __init__(self):

        # Shape configuration
        self.input_shape    = [par['num_time_steps'], 
                               par['batch_size'], 
                               par['n_input']]
        self.output_shape   = [par['num_time_steps'],
                               par['batch_size'],
                               par['n_output']]
        self.stimulus_shape = [par['num_time_steps'],
                               par['batch_size'],
                               par['num_motion_tuned']]
        self.response_shape = [par['num_time_steps'],
                               par['batch_size'],
                               par['num_motion_dirs']]
        self.fixation_shape = [par['num_time_steps'],
                               par['batch_size'],
                               par['num_fix_tuned']]

        #####################################
        # CHECK TO MAKE SURE THESE ARE NEEDED
        self.rule_shape = [par['num_time_steps'],
                           par['batch_size'],
                           par['num_rule_tuned']]
        self.mask_shape = [par['num_time_steps'],
                           par['batch_size']]
        #####################################

        # Motion and stimulus configuration
        self.motion_dirs = np.linspace(0,
                                       2 * np.pi - (2 * np.pi / par['num_motion_dirs']),
                                       par['num_motion_dirs'])

        #####################################
        # Ignoring modality for now
        # self.modality_size      = (par['num_motion_tuned'])//2
        #####################################

        # set position preference (spatial RF analogue)
        spatial_inc = 2.0 / par['num_position_locs']
        pref_positions = np.mgrid[-1 + 0.5 * spatial_inc:1.0:spatial_inc,
                                  -1 + 0.5 * spatial_inc:1.0:spatial_inc].reshape(2, -1).T

        # for dynamically-chosen stimuli: expand input
        self.pref_dir_pos = np.tile(pref_positions, 
                                    (par['num_motion_dirs'], 1))
        print(self.pref_dir_pos.shape)
        print(pref_positions.shape)
        directions = np.array([self.motion_dirs[i // pref_positions.shape[0]] 
                        for i in range(self.pref_dir_pos.shape[0])], ndmin=2).T
        print(directions.shape)
        #self.pref_dir_pos = tf.constant(np.hstack((self.pref_dir_pos, directions)), dtype=tf.float32)
        self.pref_dir_pos = np.hstack((self.pref_dir_pos, directions)).astype(np.float32)

        # final few parameters
        self.fix_time = 400
        self.rule_signal_factor = 1. if par['include_rule_signal'] else 0.

    # Von Mises distribution
    def circ_tuning(self, theta):
        ang_dist = np.angle(np.exp(1j*theta - 1j*self.pref_dir_pos[:,2]))
        return par['tuning_height']*np.exp(-0.5*(8*ang_dist/np.pi)**2)


    # 2d normal distribution, mean at pt (current sample)
    def rf_tuning(self, pts):
        pos_var = np.var(self.pref_dir_pos[:,:2])
        r = np.exp(-(pts[:,0] - self.pref_dir_pos[:,0])**2 / (pos_var * 2)) * \
            np.exp(-(pts[:,1] - self.pref_dir_pos[:,1])**2 / (pos_var * 2)) / \
            (2 * np.pi)
        return par['tuning_height'] * r

    # simultaneous direction and position tuning
    def pos_dir_tuning_tf(self, pts):
        x_variance  = tf.nn.moments(self.pref_dir_pos[:,0], axes=[0])[1]
        y_variance  = tf.nn.moments(self.pref_dir_pos[:,1], axes=[0])[1]
        direction_variance = tf.nn.moments(self.pref_dir_pos[:,2], axes=[0])[1]
        r = tf.exp(-(pts[:,0] - self.pref_dir_pos[:,0])**2 / (x_variance  * 2)) * \
            tf.exp(-(pts[:,1] - self.pref_dir_pos[:,1])**2 / (y_variance  * 2)) * \
            tf.exp(-(pts[:,2] - self.pref_dir_pos[:,2])**2 / (direction_variance * 2)) / \
            ((2 * np.pi) ** (1.5))
        return par['tuning_height'] * r


    # generate trial
    def generate_trial(self, params, cue, rule):

        par['noise_in'] = 0.05
        self.trial_info = {
            'neural_input'   : tf.Variable(tf.random_normal(self.input_shape, mean=par['input_mean'], stddev=par['noise_in'])),
            'desired_output' : tf.zeros(self.output_shape, dtype=tf.float32),
            'reward_data'    : tf.zeros(self.output_shape, dtype=tf.float32),
            'train_mask'     : tf.ones(self.mask_shape, dtype=tf.float32) }

        #print(type(self.trial_info['neural_input']))
        #print(self.trial_info['neural_input'])

        #self.trial_info['neural_input'][0,0,0].assign(tf.constant(0.))

        # if there are any rule-tuned neurons, they must receive some rule signal; 
        # update this here
        if par['num_rule_tuned'] > 0:
            rule_signal = tf.zeros((1,1,par['num_rule_tuned']))

            # create tensor with value nonzero at [0, 0, cue]
            mask_vec =  tf.cast(
                            tf.math.equal(
                                tf.range(tf.constant(self.input_shape[2])), tf.constant(self.input_shape[2] - par['num_possible_cues'] + cue)
                            ),
                            dtype = tf.float32
                        )
            scaled_mask = tf.multiply(mask_vec, tf.multiply(tf.constant(par['tuning_height']), tf.constant(self.rule_signal_factor)))
            self.trial_info['neural_input'] = tf.multiply(self.trial_info['neural_input'], scaled_mask)
            #self.trial_info['neural_input'] = self.trial_info['neural_input'][:,:,-par['num_rule_tuned']:].assi

        # generate task into trial_info
        self.task_selfexp_tf(params, cue, rule)

        # Iterate over batches
        for b in range(par['batch_size']):

            # Designate timings
            respond_time    = np.where(np.sum(self.trial_info['desired_output'][:,b,:-1], axis=1) > 0)[0]

            fix_time        = list(range(respond_time[0])) if len(respond_time) > 0 else [-1]
            respond_time    = respond_time if len(respond_time) > 0 else [-1]

            # Designate responses
            correct_response    = np.where(self.trial_info['desired_output'][respond_time[0],b,:]==1)[0]
            incorrect_response  = np.where(self.trial_info['desired_output'][respond_time[0],b,:-1]==0)[0]

            # Build reward data
            self.trial_info['reward_data'][fix_time,b,:-1] = par['fix_break_penalty']
            self.trial_info['reward_data'][respond_time,b,correct_response] = par['correct_choice_reward']
            for i in incorrect_response:
                self.trial_info['reward_data'][-1,b,-1] = par['wrong_choice_penalty']

            # Penalize fixating throughout entire trial if response was required
            '''if not self.trial_info['desired_output'][-1,b,-1] == 1:
                self.trial_info['reward_data'][-1,b,-1] = par['fix_break_penalty']
            else:
                self.trial_info['reward_data'][-1,b,-1] = par['correct_choice_reward']'''

        # Returns the task name and trial info
        return cue, self.trial_info['neural_input'], self.trial_info['desired_output'], \
            self.trial_info['train_mask'], self.trial_info['reward_data']




    # assume one trial at a time (e.g. batch size of 1)
    def task_selfexp_tf(self, params, cue=0, rule=np.array([np.pi, 1]), moving=False):

        """ 
        Trials for `self-experimenting' RNN (e.g. a net that selects/constructs the examples
        upon which it trains).

        Task structure:
        --------------
        ITI               - 200ms
        action cue        - 200ms
        sample + response - 400ms

        e.g. trial[0:action_cue_on]       -> ITI
             trial[action_cue_on:stim_on] -> action cue (fixation)
             trial[stim_on:]              -> stimulus + response

        """

        # verify arguments
        num_cues = par['num_possible_cues']

        # set task parameters (time of stimulus onset, etc.)
        action_cue_on  = par['dead_time']
        stim_on        = tf.floordiv(2 * action_cue_on + tf.random_normal((1,), 0, 50), tf.constant(float(par['dt'])))

        # unpack params, use to create stimulus tuning (depending on rule);
        # locs carries the location of the moving stimulus, updated through
        # time; stim_input carries what the input neurons receive, which
        # reflects that motion through space + time
        angle = tf.angle(tf.complex(params[2], params[3]))[0]
        angle = tf.cond(tf.less(angle, tf.constant(0.)), lambda: tf.add(angle, tf.constant(np.pi)), lambda: angle)
        locs  = tf.ones(((par['num_time_steps'] - stim_on)[0], params.shape[0]))
        locs = locs * tf.transpose(params)

        """
        if moving:
            locs  = tf.concat(concat_dim = 1, values = [locs, tf.Variable(range(locs.shape[0]))])
            locs  = tf.apply_along_axis(
                        lambda a: 
                            np.array([a[0] + (a[2] * a[-1]), 
                                      a[1] + (a[3] * a[-1]),
                                      angle]) 
                    )
        """
        angles = tf.ones(((par['num_time_steps'] - stim_on)[0], 1)) * angle
        print(angles.shape)
        print(locs.shape)

        # tack on angle as third column
        locs = tf.concat(axis = 1, 
                         values = [locs[:,:2], 
                                   angles 
                                   ]
                        )
        stim_input = self.pos_dir_tuning(locs)
        print("STIMINPUT:",stim_input.shape)
        #print(0/0)

        # set up mask; later, set zeros
        self.trial_info['train_mask'] = np.ones(self.mask_shape)

        # use stim, rule, and cue to compute response

        # DIRECTION: if category match, go right, else left
        if cue == 0:
            print(type(rule))
            a = tf.div(tf.subtract(rule[0,0], angle), rule[0, 1])
            print("a.shape", a.shape)
            #resp = tf.cond(tf.greater((rule[0,0] - angle) / rule[0, 1], tf.constant(0.)), lambda: [0, 0, 1], lambda: [0, 1, 0])
            resp = tf.cond(tf.greater(a, tf.constant(0.)), lambda: [0, 0, 1], lambda: [0, 1, 0])
            """if (rule[0,0] - angle) / rule[0,1] > 0:
                resp = [0, 0, 1]
            else:
                resp = [0, 1, 0]"""

        # POSITION: if category match, go right, else left
        elif cue == 1:
            print(type(rule))
            print(rule[0,0])
            print(params.shape)
            print(type(params[0]))
            a = tf.div(tf.multiply(rule[0,0], params[0]), rule[0,1])
            print("a.shape", a.shape)
            print("a", a)
            #resp = tf.cond(tf.greater((rule[0,0] * params[0]) / rule[0,1], tf.constant(0.)), lambda: [0, 0, 1], lambda: [0, 1, 0])
            resp = tf.cond(tf.greater(a[0], tf.constant(0.)), lambda: [0, 0, 1], lambda: [0, 1, 0])
            """if ((rule[0,0] * params[0]) - params[1]) / rule[0,1] > 0:
                resp = [0, 0, 1]
            else:
                resp = [0, 1, 0]"""

        # TODO: when cue == 2 (e.g. rule dependent on position and direction), what to do?
        resp = tf.cast(resp, tf.float32)

        # set up actual trial
        for b in range(par['batch_size']):

            print(stim_on.shape)
            print(type(stim_on))
            print(type(stim_on[0]))

            # set input
            zero_template = tf.zeros(self.input_shape)

            # time masks
            ITI_mask  = tf.cast(
                            tf.less(
                                tf.range(self.input_shape[0]), tf.constant(action_cue_on)
                            ), dtype=tf.int32
                        )
            stim_mask = tf.cast(
                            tf.greater_equal(
                                tf.range(self.input_shape[0]), tf.cast(stim_on[0], dtype=tf.int32)
                            ), dtype=tf.int32
                        )
            cue_mask  = tf.cast(
                            tf.logical_and(
                                tf.greater(
                                    tf.range(self.input_shape[0]), ITI_mask
                                ), 
                                tf.less_equal(
                                    tf.range(self.input_shape[0]), stim_mask
                                )
                            ), dtype=tf.int32
                        )

            ITI_mask_neg = tf.cast(tf.logical_not(tf.cast(ITI_mask, dtype=tf.bool)), dtype=tf.int32)
            stim_mask_neg = tf.cast(tf.logical_not(tf.cast(stim_mask, dtype=tf.bool)), dtype=tf.int32)
            cue_mask_neg = tf.cast(tf.logical_not(tf.cast(cue_mask, dtype=tf.bool)), dtype=tf.int32)
            stim_mask = tf.cast(stim_mask, dtype=tf.float32)
            cue_mask = tf.cast(cue_mask, dtype=tf.float32)

            # neuron masks now
            motion_tuned_mask = tf.cast(tf.less(tf.range(self.input_shape[2]), tf.constant(par['num_motion_tuned'])), dtype=tf.int32)
            fix_tuned_mask    = tf.cast(
                                    tf.logical_and(
                                        tf.greater(
                                            tf.range(self.input_shape[2]), motion_tuned_mask
                                        ), 
                                        tf.less_equal(
                                            tf.range(self.input_shape[2]), tf.constant(par['num_motion_tuned'] + par['num_fix_tuned'])
                                        )
                                    ), dtype=tf.int32
                                )

            motion_tuned_mask_neg = tf.cast(tf.logical_not(tf.cast(motion_tuned_mask, dtype=tf.bool)), dtype=tf.float32)
            fix_tuned_mask_neg = tf.cast(tf.logical_not(tf.cast(fix_tuned_mask, dtype=tf.bool)), dtype=tf.float32)

            # set cue + fixation neurons
            indices = [cue] * (tf.cast(stim_on[0], tf.int32) - (action_cue_on // par['dt']))
            cue_vec = tf.expand_dims(tf.one_hot(indices, par['num_possible_cues']), 1)
            cue_vec = tf.concat([tf.zeros([action_cue_on // par['dt'], 1, par['num_possible_cues']]), cue_vec], 0)
            cue_vec = tf.concat([cue_vec, tf.zeros([self.input_shape[0] - tf.cast(stim_on[0], tf.int32), 1, par['num_possible_cues']])], 0)
            cue_vec = tf.concat([tf.zeros([self.input_shape[0], 1, par['num_fix_tuned']]), cue_vec], 2)
            print(cue_vec.shape)

            #cue_through_time = tf.tile(tf.expand_dims())

            # set ITI; should be [10, 1, 512] in shape
            ITI_through_time = tf.zeros([action_cue_on // par['dt'], 1, par['num_motion_tuned']])
            during_cue       = tf.zeros([tf.cast(stim_on[0], tf.int32) - action_cue_on // par['dt'], 1, par['num_motion_tuned']])
            #pre_stimulus_input = tf.concat([ITI_through_time, during_cue], 0)
            #print(pre_stimulus_input.shape)

            # set stimulus; should be [100, 1, 512] in shape AFTER CONCATENATING ITI
            stim_input_through_time = tf.tile(tf.expand_dims(stim_input, 0), [tf.cast(self.input_shape[0] - stim_on[0], tf.int32), 1])
            stim_input_through_time = tf.expand_dims(stim_input_through_time, 1)
            stim_input_through_time = tf.concat([during_cue, stim_input_through_time], axis=0)
            print(stim_input_through_time.shape)

            # set noise
            noise_to_add = tf.random_normal([self.input_shape[0] - (action_cue_on // par['dt']), self.input_shape[1], self.input_shape[2] - par['num_fix_tuned'] - par['num_rule_tuned']], mean=par['input_mean'], stddev=par['noise_in'])
            print(noise_to_add.shape)

            stim_input_through_time = tf.add(stim_input_through_time, noise_to_add)
            print(stim_input_through_time.shape)

            # concatenate all together to form one input
            input_through_time = tf.concat([tf.concat([ITI_through_time, stim_input_through_time], 0), cue_vec], 2)
            print(input_through_time.shape)

            # set response; should be [100, 1, 3] in shape
            resp_through_time = tf.tile(tf.expand_dims(resp, 0), [tf.cast(self.output_shape[0] - stim_on[0], tf.int32), 1])
            resp_through_time = tf.expand_dims(resp_through_time, 1)
            resp_padding      = tf.zeros([tf.cast(stim_on[0], tf.int32), 1, self.output_shape[2]], dtype=tf.float32)
            resp_through_time = tf.concat([resp_padding, resp_through_time], 0)
            print(resp_through_time.shape)

            # set response
            self.trial_info['desired_output'] = resp_through_time

            # set input
            self.trial_info['neural_input'] = input_through_time
            
            # set mask
            self.trial_info['train_mask'] = tf.concat([tf.zeros([action_cue_on // par['dt'], 1]), tf.ones([self.input_shape[0] - (action_cue_on // par['dt']), 1])], 0)


        return self.trial_info