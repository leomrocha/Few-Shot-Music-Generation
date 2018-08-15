
from torch import nn

from models.pt_tcn import TemporalConvNet
from models.pt_model import PyTorchModel


class SimpleTCN(nn.Module):
    def __init__(self, in_size, embd_size, out_size, num_channels, kernel_size, dropout, future=1):
        super(SimpleTCN, self).__init__()
        self.embeddings = nn.Embedding(in_size, embd_size)
        self.tcn = TemporalConvNet(embd_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], out_size)
        # self.relu = nn.ReLu()
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def init_hidden(self):
        self.tcn.init_weights()

    def forward(self, x):
        # print("tcn x shape = ",x.shape)
        output = self.embeddings(x)
        # print("tcn out shape = ",output.shape)
        output = self.tcn(output.transpose_(1, 2)).transpose_(1, 2)
        # print("tcn out shape = ",output.shape)
        output = self.linear(output)
        # output = self.relu(output)
        output = self.sig(output)
        output = self.softmax(output)
        #print("out shape = ", output.shape)
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
        self._hidden_size = self._config['hidden_size']
        # self._num_channels = [self._input_size, self._config["embedding_size"], self._config["embedding_size"]]
        self._num_channels = [self._embd_size] + [self._hidden_size] * self._n_layers

        self.model = SimpleTCN(in_size=self._input_size, embd_size=self._embd_size,
                               out_size=self._input_size,
                               num_channels=self._num_channels, future=self._time_steps,
                               kernel_size=self._kernel_size,
                               dropout=0
                               )

        self.model.to(self.device)
