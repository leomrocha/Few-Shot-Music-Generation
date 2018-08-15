import torch
import torch.nn as nn
from torch import tensor
import torch.nn.functional as F
from models.pt_model import PyTorchModel


class PlasticLayer(nn.Module):
    """
    PlasticLayer implements hebbian basic and OJA's rules for learning in a fully connected manner.
    This class is mostly based on the differentiable-plasticity work from Uber here:
    https://github.com/uber-research/differentiable-plasticity hence the same license as that repo is applied to this file
    """
    def __init__(self, embedding_size, plasticity="hebbian"):
        """
        :param embedding_size: size of the layer in neurons
        :param plasticity: [nonplastic|hebbian|oja]
        """
        super(PlasticLayer, self).__init__()
        self.embedding_size = embedding_size
        self.plasticity = plasticity
        # Notice that the vectors are row vectors, and the matrices are transposed wrt the usual order,
        #  following apparent pytorch conventions
        # Each *column* of w targets a single output neuron
        # The matrix of fixed (baseline) weights
        self.w = tensor(.01 * torch.randn(embedding_size, embedding_size), requires_grad=True)
        # The matrix of plasticity coefficients
        self.alpha = tensor(.01 * torch.randn(embedding_size, embedding_size), requires_grad=True)
        # The eta coefficient is learned
        self.eta = tensor(.01 * torch.ones(1), requires_grad=True)
        if self.plasticity is 'nonplastic':
            self.zero_diag_alpha()  # No plastic autapses
        elif self.plasticity is "hebbian":
            pass
        elif self.plasticity is "oja":
            pass
        else:
            raise ValueError("Wrong network type!")

    def forward(self, input, yin, hebb):
        # Run the network for one timestep
        if self.plasticity is 'hebbian':
            yout = F.tanh(yin.mm(self.w + torch.mul(self.alpha, hebb)) + input)
            # bmm used to implement outer product with the help of unsqueeze (i.e. added empty dimensions)
            hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(yin.unsqueeze(2), yout.unsqueeze(1))[0]
        elif self.plasticity is 'oja':
            yout = F.tanh(yin.mm(self.w + torch.mul(self.alpha, hebb)) + input)
            # Oja's rule.  TODO check that this is OK in the dimensions ... I might have messed something
            hebb = hebb + self.eta * torch.mul((yin.unsqueeze(2) - torch.mul(hebb, yout.unsqueeze(1))),
                                               yout.unsqueeze(1))[0]
        elif self.plasticity is 'nonplastic':
            yout = F.tanh(yin.mm(self.w) + input)
        else:
            raise ValueError("Wrong network type!")
        return yout, hebb

    def initial_zero_state(self):
        return tensor(torch.zeros(1, self.embedding_size))

    def initial_zero_hebb(self):
        return tensor(torch.zeros(self.embedding_size, self.embedding_size))

    def zero_diag_alpha(self):
        # Zero out the diagonal of the matrix of alpha coefficients: no plastic autapses
        self.alpha.data -= torch.diag(torch.diag(self.alpha.data))


class PlasticFCNetwork(nn.Module):
    def __init__(self, embedding_size, plasticity="hebbian"):
        """
        :param embedding_size: size of the layer in neurons
        :param plasticity: [nonplastic|hebbian|oja]
        """
        super(PlasticFCNetwork, self).__init__()
        # TODO


class PlasticFCBaseline(PyTorchModel):
    """
    TCNBaseline
    """
    def __init__(self, config):
        super(PlasticFCBaseline, self).__init__(config)
        # self._hidden_size = self._config['hidden_size']
        self._n_layers = self._config['n_layers']
        self._kernel_size = self._config['kernel_size']
        self._hidden_size = self._config['hidden_size']
        # self._num_channels = [self._input_size, self._config["embedding_size"], self._config["embedding_size"]]
        self._num_channels = [self._embd_size] + [self._hidden_size] * self._n_layers

        self.model = PlasticFCNetwork(in_size=self._input_size, embd_size=self._embd_size,
                               out_size=self._input_size,
                               num_channels=self._num_channels, future=self._time_steps,
                               kernel_size=self._kernel_size,
                               dropout=0
                               )

        self.model.to(self.device)
