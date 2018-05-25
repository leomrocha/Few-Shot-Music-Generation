import os
from torch import nn
import pprint
from models.base_model import BaseModel

PP = pprint.PrettyPrinter(depth=6)



class PyTorchModel(BaseModel):
    """docstring for PyTorchModel.BaseModel"""
    def __init__(self, arg):
        super(PyTorchModel,BaseModel.__init__())
        self.arg = arg

    def train(self, episode):
        """Train model on episode.

        Args:
            episode: Episode object containing support and query set.
        """
        raise NotImplementedError()

    def eval(self, episode):
        """Evaluate model on episode.

        Args:
            episode: Episode object containing support and query set.
        """
        raise NotImplementedError()

    def sample(self, support_set, num):
        """Sample a sequence of size num conditioned on support_set.

        Args:
            support_set (numpy array): support set to condition the sample.
            num: size of sequence to sample.
        """
        raise NotImplementedError()

    def save(self, checkpt_path):
        """Save model's current parameters at checkpt_path.

        Args:
            checkpt_path (string): path where to save parameters.
        """
        raise NotImplementedError()

    def recover_or_init(self, init_path):
        """Recover or initialize model based on init_path.

        If init_path has appropriate model parameters, load them; otherwise,
        initialize parameters randomly.
        Args:
            init_path (string): path from where to load parameters.
        """
        raise NotImplementedError()
