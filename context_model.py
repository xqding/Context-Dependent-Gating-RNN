import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextModel(nn.Module):
    '''Given a time series of stimulus, action, and reward, predict
       which context/task the time serier come from.'''
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size):
        super(ContextModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.hidden = self.init_hidden()
        self.hidden2output = nn.Linear(hidden_dim, output_dim)

    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_dim).cuda(),
                torch.zeros(1, self.batch_size, self.hidden_dim).cuda())

    def forward(self, inputs):
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        output = self.hidden2output(self.hidden[0])
        output_score = F.log_softmax(output, dim = -1)
        return output_score
    
