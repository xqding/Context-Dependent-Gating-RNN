# Author: NY Masse + GD Grant (modified by M Rosen)
import numpy as np
from parameters import par
import tensorflow as tf


class DynamicStimulus:

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
        directions = np.array([self.motion_dirs[i // pref_positions.shape[0]] 
                        for i in range(self.pref_dir_pos.shape[0])], ndmin=2).T
        self.pref_dir_pos = np.hstack((self.pref_dir_pos, directions)).astype(np.float32)

        # final few parameters; remnant of MultiStimulus file
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
    def pos_dir_tuning(self, pts):
        x_variance  = np.var(self.pref_dir_pos[:,0])
        y_variance  = np.var(self.pref_dir_pos[:,1])
        direction_variance = np.var(self.pref_dir_pos[:,2])

        rs = []
        for i in range(pts.shape[0]):
            rs.append(np.exp(-(pts[i,0] - self.pref_dir_pos[:,0])**2 / (x_variance  * 2)) * \
                np.exp(-(pts[i,1] - self.pref_dir_pos[:,1])**2 / (y_variance  * 2)) * \
                np.exp(-(pts[i,2] - self.pref_dir_pos[:,2])**2 / (direction_variance * 2)) / \
                ((2 * np.pi) ** (1.5)) * par['tuning_height'])

        return np.array(rs)

    # generate trial
    def generate_trial(self, params, cue, rule):

        #par['noise_in'] = 0.02
        self.trial_info = {
            'neural_input'   : np.random.normal(par['input_mean'], par['noise_in'], size=self.input_shape).astype(np.float32),
            'desired_output' : np.zeros(self.output_shape, dtype=np.float32),
            'reward_data'    : np.zeros(self.output_shape, dtype=np.float32),
            'train_mask'     : np.ones(self.mask_shape, dtype=np.float32) }

        # if there are any rule-tuned neurons, they must receive some rule signal; 
        # update this here
        if par['num_rule_tuned'] > 0:
            rule_signal = np.zeros((1,1,par['num_rule_tuned']))
            rule_signal[0, 0, cue] = par['tuning_height']
            self.trial_info['neural_input'][:, :, -par['num_rule_tuned']:] += rule_signal*self.rule_signal_factor

        # generate task into trial_info
        self.task_selfexp(params, cue, rule)

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
            #self.trial_info['reward_data'][fix_time,b,:-1] = par['fix_break_penalty']
            self.trial_info['reward_data'][respond_time,b,correct_response] = par['correct_choice_reward']
            for i in incorrect_response:
                self.trial_info['reward_data'][-1,b,-1] = par['wrong_choice_penalty']

            # Penalize fixating throughout entire trial if response was required
            if not self.trial_info['desired_output'][-1,b,-1] == 1:
                self.trial_info['reward_data'][-1,b,-1] = par['fix_break_penalty']
            else:
                self.trial_info['reward_data'][-1,b,-1] = par['correct_choice_reward']

        # Returns the task name and trial info
        return [np.float32(cue), np.float32(self.trial_info['neural_input']), np.float32(self.trial_info['desired_output']), \
            np.float32(self.trial_info['train_mask']), np.float32(self.trial_info['reward_data'])]

    def task_selfexp(self, params, cue=0, rule=np.array([np.pi, 1])):
        """ 
        Trials for `self-experimenting' RNN (e.g. a net that selects/constructs the examples
        upon which it trains).

        Task structure:
        --------------
        ITI               - approx 200ms
        action cue        - approx 200ms
        sample + response - approx 400ms

        e.g. trial[0:action_cue_on]       -> ITI
             trial[action_cue_on:stim_on] -> action cue (fixation?)
             trial[stim_on:]              -> stimulus + response (no delay)

        """

        # verify arguments
        num_cues = par['num_possible_cues']

        # set task parameters (time of stimulus onset, etc.)
        action_cue_on  = par['dead_time'] // par['dt']
        stim_on        = action_cue_on + int((action_cue_on + (np.random.normal(0, 50, 1).astype(np.float32) // par['dt']))[0])

        # unpack params, to calculate angle; locs stores the following:
        #   locs[i,0] -- x-position of stimulus at time i
        #   locs[i,1] -- y-position of stimulus at time i
        #   locs[i,2] -- angle of motion of stimulus at time i
        angle = np.angle(params[2] + 1j * params[3]).astype(np.float32)
        if angle < 0:
            angle = angle + np.pi

        locs = np.tile(params[:2], (int(par['num_time_steps'] - stim_on), 1))
        b = [angle] * locs.shape[0]
        locs = np.hstack((locs, np.array(b)[:, np.newaxis]))

        stim_input = self.pos_dir_tuning(locs)

        # set up mask; later, set zeros
        self.trial_info['train_mask'] = np.ones(self.mask_shape)

        # DIRECTION: if category match, go right, else left
        if cue == 0:
            if (rule[0,0] - angle) / rule[0,1] > 0:
                resp = [0, 1, 0]
            else:
                resp = [1, 0, 0]

        # POSITION: if category match, go right, else left
        elif cue == 1:
            if ((rule[0,0] * params[0]) - params[1]) / rule[0,1] > 0:
                resp = [0, 1, 0]
            else:
                resp = [1, 0, 0]

        # TODO: when cue == 2 (e.g. rule dependent on position and direction), what to do?

        # set up actual trial
        for b in range(par['batch_size']):

            # set input
            self.trial_info['neural_input'][0:action_cue_on      , b, :                       ] = 0
            self.trial_info['neural_input'][stim_on:             , b, :par['num_motion_tuned']] += stim_input

            """
                QUESTION: how do we want to deal with fixation? Are we going to require it here? Also, can we clarify
                          exactly how action cue (denoting which task is currently active) is to be passed? Are we 
                          using rule-tuned neurons for this?
            """
            ##           
            #self.trial_info['neural_input'][action_cue_on:stim_on, b, par['num_motion_tuned']:par['num_motion_tuned'] \
            #                                                            + par['num_fix_tuned']] = par['tuning_height']

            # set response
            self.trial_info['desired_output'][stim_on:, b, :] = resp
            
            # set mask; we don't want to train on activity during ITI
            self.trial_info['train_mask'][:action_cue_on] = 0


        return self.trial_info





