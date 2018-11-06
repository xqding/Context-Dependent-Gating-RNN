import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from stimulus import *
import sys


class TrialData(Dataset):
    "Trial data including stimulus, action, and reward"
    def __init__(self, stim, num_tasks, num_time_steps):
        super(TrialData, self).__init__()
        self.stim = stim
        self.num_tasks = num_tasks
        self.num_time_steps = num_time_steps
        
    def __len__(self):
        return self.num_time_steps

    def __getitem__(self, idx):
        idx_time_step = idx + 1
        inputs = []
        task_id = []
        for tid in range(15):            
            ## generate stimulus, action, and reward        
            name, stim_in, y_hat, mk, reward = self.stim.generate_trial(tid)
            _, batch_size, dim_input = stim_in.shape
            num_actions = y_hat.shape[-1]
            
            ## stimulus input from step 0 to idx_time_step
            tmp_stim_in = stim_in[0:idx_time_step, :, :]

            ## action from step 0 to (idx_time_step -1)    
            tmp_action = np.float64(reward[0:(idx_time_step-1)] == 0)
            #assert(np.all(np.sum(tmp_action == 1, -1)))

            ## make sure all the rewards from step 0 to (idx_time_step-1) are zeros
            #assert(np.all(reward[0:(idx_time_step-1)][tmp_action!=0] == 0))

            ## the reward from the last step is not zeros; find out actions that have
            ## nonzero reward and randomly sample one from them as the last step action
            non_zero_last_action = np.where(reward[idx_time_step] != 0)[1].reshape((batch_size,-1))
            last_step_action =  np.apply_along_axis(np.random.choice, 1, non_zero_last_action)
            last_step_action = np.identity(num_actions)[last_step_action]

            ## combine actions from step 0 to (idx_time_step-1) with the last step action
            tmp_action = np.vstack([tmp_action, last_step_action[np.newaxis, :, :]])

            ## pull out rewards from step 0 to idx_time_step based on the actions taken
            tmp_reward = reward[0:idx_time_step][tmp_action != 0].reshape((idx_time_step, batch_size))

            ## combine stimulus, action, and reward together
            tmp_input = np.dstack([tmp_stim_in, tmp_action, tmp_reward[:,:,np.newaxis]])
            inputs.append(tmp_input)
            task_id += [tid for i in range(batch_size)]
        
        inputs = torch.from_numpy(np.hstack(inputs))
        task_id = torch.tensor(task_id)
        
        return inputs.float(), task_id.long()
    
