import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
from collections import deque, defaultdict
import time

from stimulus import *
from load_data import *
from context_model import *

num_tasks = 15
num_time_steps = 99
stim = MultiStimulus()
data = TrialData(stim, num_tasks, num_time_steps)
input, output = data[2]
_, batch_size, input_dim = input.shape
hidden_dim = 30
output_dim = num_tasks

def collate_fn(data):
    input = [data[i][0] for i in range(len(data))]
    target = [data[i][1] for i in range(len(data))]    
    return input, target

load_batch_size = 8
dataloader = DataLoader(data, batch_size = load_batch_size, collate_fn = collate_fn, num_workers = 4)
model = ContextModel(input_dim, hidden_dim, output_dim, batch_size)
model.cuda()
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

accuracy = defaultdict(deque)
for idx_time in range(1, 100):
    accuracy[idx_time].append(0.0)
    
for epoch in range(100):
    for idx, data in enumerate(dataloader):
        for i in range(len(data[0])):
            idx_time = idx * load_batch_size + i + 1
            ## load data            
            input, target = data[0][i], data[1][i]
            input, target = input.cuda(), target.cuda()
            
            ## zero derivatives
            model.zero_grad()
        
            ## initialize hidden units of lstm
            model.hidden = model.init_hidden()

            ## forward
            score = model(input)
            score = score.reshape(score.shape[1:])
            loss = loss_function(score, target)

            ## backward
            loss.backward()
            optimizer.step()

            ## collect statistics
            predict = torch.argmax(score, -1)
            accuracy[idx_time].append(torch.mean((predict == target).float()))
            if len(accuracy[idx_time]) > 2:
                accuracy[idx_time].popleft()

            if (idx_time + 1) % 10 == 0:
                print("Epoch: {:>4d}, idx_time: {:>3d}, loss: {:5.2f}, previous_accuracy: {:5.2f}, current_accuracy: {:5.2f}".
                      format(epoch, idx_time, loss.item(), accuracy[idx_time][0], accuracy[idx_time][1]))
