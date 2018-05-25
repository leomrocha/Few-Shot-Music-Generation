
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
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(SimpleTCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
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


    def evaluate(self, episode):
        query_set = episode.query
        X, Y = convert_tokens_to_input_and_target(query_set)
        self.eval()
        x = Variable(torch.Tensor(X)).cuda()
        y = Variable(torch.Tensor(Y)).cuda()
        output = self(x.unsqueeze(0)).squeeze(0)
        print(output)
        loss = -torch.trace(torch.matmul(y, torch.log(output).float().t()) +
                            torch.matmul((1-y), torch.log(1-output).float().t()))
        print("loss = ", loss)
        return loss
        # total_loss += loss.data[0]
        # count += output.size(0)
        # eval_loss = total_loss / count
        # print("Validation/Test loss: {:.5f}".format(eval_loss))
        # return eval_loss

class TCN_baseline(PyTorchModel):
    """docstring for TCN_baseline."""
    def __init__(self, arg):
        super(TCN_baseline, self).__init__()
        self.arg = arg
