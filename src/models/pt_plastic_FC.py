import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from models.pt_model import PyTorchModel
from models.base_model import convert_tokens_to_input_and_target


class TCNPlasticFC(nn.Module):
    def __init__(self, in_size, out_size, num_channels, kernel_size, dropout, future=1):
        super(TCNPlasticFC, self).__init__()
        self.tcn = TemporalConvNet(in_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], out_size)
        # self.relu = nn.ReLu()
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def init_hidden(self):
        self.tcn.init_weights()

    def forward(self, x):
        output = self.tcn(x.transpose_(1, 2)).transpose_(1, 2)
        #print("tcn out shape = ",output.shape)
        output = self.linear(output)
        # output = self.relu(output)
        output = self.sig(output)
        output = self.softmax(output)
        #print("out shape = ", output.shape)
        return output


class TCNPlasticFCBaseline(PyTorchModel):
    """
    PlasticFCBaseline
    """
    def __init__(self, config):
        super(TCNPlasticFCBaseline, self).__init__(config)
        # self._hidden_size = self._config['hidden_size']
        self._n_layers = self._config['n_layers']
        # self._start_word = self._config['input_size']
        self._kernel_size = self._config['kernel_size']

        self._num_channels = [self._input_size, self._config["embedding_size"], self._config["embedding_size"]]
        # self._num_channels = self._config["embedding_size"]
        self.model = TCNPlasticFC(in_size=self._input_size, out_size=self._input_size,
                               num_channels=self._num_channels, future=self._time_steps,
                               kernel_size=self._kernel_size,
                               dropout=0
                               )

        self.model.to(self.device)
