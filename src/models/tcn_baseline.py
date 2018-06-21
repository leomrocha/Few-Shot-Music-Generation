
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from models.tcn import TemporalConvNet
from models.pt_model import PyTorchModel
from models.base_model import convert_tokens_to_input_and_target


class SimpleTCN(nn.Module):
    def __init__(self, in_size, out_size, num_channels, kernel_size, dropout, future=1):
        super(SimpleTCN, self).__init__()
        self.tcn = TemporalConvNet(in_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        # self.relu = nn.ReLu()
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        #output = self.tcn(x)
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        output = self.linear(output)
        # output = self.relu(output)
        output = self.sig(output)
        output = self.softmax(output)
        return output

class TCNBaseline(PyTorchModel):
    """
    TCNBaseline
    """
    def __init__(self, config):
        super(TCNBaseline, self).__init__(config)
        # self._hidden_size = self._config['hidden_size']
        self._n_layers = self._config['n_layers']
        self._kernel_size = self._config['kernel_size']

        self.model = SimpleTCN(in_size=self._input_size, out_size=self._input_size,
                               num_channels=self._n_layers, future=self._time_steps,
                               dropout=0
                               )
        self.model.to(self.device)
