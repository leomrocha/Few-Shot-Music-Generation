
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Softmax
import torch.nn.functional as F
import torch.optim as optim

from models.pt_model import PyTorchModel


class SimpleRecurrentSequence(nn.Module):

    # Options that can be used for the recurrent module, this makes the class more general without any change
    # as pytorch has a nice unified interface for the 3 classes
    module_dict = {
                   "RNN": nn.RNN,
                   "LSTM": nn.LSTM,
                   "GRU": nn.GRU,
                   }

    def __init__(self, in_size, embd_size, hid_size, out_size, layers, future=0, bias=True, dropout=0, batch_size=1,
                 bidirectional=False, cell="LSTM", batch_first=False):
        super(SimpleRecurrentSequence, self).__init__()
        self.in_size = in_size
        self.hid_size = hid_size
        self.out_size = out_size
        self.layers = layers
        self.future = future
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.CELL_TYPE = SimpleRecurrentSequence.module_dict[cell]
        self.embeddings = nn.Embedding(in_size, embd_size)
        self.rnn = self.CELL_TYPE(input_size=in_size, hidden_size=hid_size, num_layers=layers, bias=bias,
                                  dropout=dropout, bidirectional=bidirectional, batch_first=batch_first)
        self.linear = nn.Linear(hid_size, out_size)
        self.softmax = Softmax()  # (dim=self.out_size)
        #self.hidden = self.init_hidden()  #TODO fix the issue with cuda access here
        self.hidden = None

    def forward(self, data, future=0):
        outputs = []
        print("inshape: ", data.shape)
        embd_data = self.embeddings(data)
        print("embd_data: ", embd_data.shape)
        # print(self.hidden)
        # TODO FIXME here there is a size mismatch RuntimeError: input must have 3 dimensions, got 2
        out, self.hidden = self.rnn(embd_data, self.hidden) if self.hidden is not None else self.rnn(data)
        print("shapes: ", out.shape, self.hidden[0].shape)
        out = self.linear(out)
        out = self.softmax(out)
        outputs += [out]
        output = out  # .view(-1,999,1)
        if future > 0:
            # I want to avoid that it actually changes the current real state
            # for the next step to avoid future knowledge during training
            hidd = (e.clone() for e in self.hidden)
        for i in range(future-1):  # if we should predict the future
            out, hidd = self.rnn(out, hidd)
            out = self.linear(out)
            outputs += [out]
        outputs = torch.stack(outputs, 1).squeeze(1)
        return outputs

    def init_hidden(self, device):
        n_directions = 2 if self.bidirectional else 1

        return (torch.zeros(self.layers, self.batch_size, self.hid_size).to(device),
                torch.zeros(self.layers, self.batch_size, self.hid_size).to(device))


class RNNBaseline(PyTorchModel):
    """
    RNNBaseline
    Is able to create RNN, GRU and LSTM from configuration.
    """
    def __init__(self, config):
        super(RNNBaseline, self).__init__(config)
        self._hidden_size = self._config['hidden_size']
        self._n_layers = self._config['n_layers']
        self._cell_type = "LSTM"  # TODO add this from configuration

        self.model = SimpleRecurrentSequence(in_size=self._input_size, embd_size=self._embd_size,
                                            hid_size=self._hidden_size,
                                            out_size=self._input_size, layers=self._n_layers,
                                            future=self._time_steps, bias=True,
                                            dropout=0, batch_size=1,
                                            bidirectional=False, cell=self._cell_type,
                                            batch_first=False
                                            )
        self.model.to(self.device)
