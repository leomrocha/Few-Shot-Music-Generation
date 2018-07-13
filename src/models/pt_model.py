import os
import torch
from torch import nn
import pprint
from models.base_model import BaseModel, convert_tokens_to_input_and_target
import numpy as np

PP = pprint.PrettyPrinter(depth=6)

class OneHot(nn.Module):
    # TODO module to use the embedding it directly in the network graph
    # this should make it more time-efficient than scatter_ implementation
    pass

class PyTorchModel(BaseModel):
    """
    docstring for PyTorchModel.BaseModel
    """
    def __init__(self, config):
        super(PyTorchModel,self).__init__(config)
        self._start_word = self._config['input_size']
        self._input_size = self._config['input_size'] + 1

        self._time_steps = self._config['max_len']
        self._embd_size = self._config['embedding_size']
        self._lr = self._config['lr']
        self._max_grad_norm = self._config['max_grad_norm']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # TODO change loss criterion
        self.criterion = nn.MSELoss() #this loss does not have backward
        #criterion = nn.CrossEntropyLoss() # FIXME this loss does not work with the current setup

    def _to_one_hot(self, data):
        return one_hot(data, self._input_size, self.device)

    def _encode(self, data):
        return self.encoder(data)

    def train(self, episode):
        """Train model on episode.

        Args:
            episode: Episode object containing support and query set.
        """
        # format input data
        X, Y = convert_tokens_to_input_and_target(
            episode.support, self._start_word)
        X2, Y2 = convert_tokens_to_input_and_target(
            episode.query, self._start_word)
        X = np.concatenate([X, X2])
        X = torch.from_numpy(X).to(self.device)
        Y = np.concatenate([Y, Y2])
        Y = torch.from_numpy(Y).to(self.device)
        X.grad = None
        Y.grad = None
        # create embedding
        X = self._encode(X)
        Y = self._encode(Y)
        #train
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr)
        self.model.hidden = self.model.init_hidden(self.device) #reset state to avoid interference between different elements
        self.model.zero_grad()
        out = self.model(X, future=self._time_steps)
    #     print(out.shape)
        loss = self.criterion(out, Y)
    #     print(type(loss))
        loss.backward()
        optimizer.step()
        X = Y = X2 = Y2 = None
        try:
            torch.cuda.empty_cache()
        except:
            print("error emptying memory")
            pass
        return loss

    def eval(self, episode):
        """Ignore support set and evaluate only on query set."""
        # print(episode.query.shape)
        X, Y = convert_tokens_to_input_and_target(episode.query, self._start_word)
        X = torch.from_numpy(X).to(self.device)
        Y = torch.from_numpy(Y).to(self.device)
        X.grad = None
        Y.grad = None
        # create embedding
        X = self._encode(X)
        Y = self._encode(Y)
        # print(X.shape, Y.shape)
        out = self.model(X)
        # print(out.shape)
        loss = self.criterion(out, Y)
        return loss

    def sample(self, support_set, num):
        """Ignore support set for sampling."""
        pred_words = []
        # word = self._start_word
        #
        # state = self._sess.run(self._cell.zero_state(1, tf.float32))
        # x = np.zeros((1, self._time_steps))
        # self.model.forward(support_set, future=num)
        # # TODO
        # pred_words = TODO
        return pred_words

    def save(self, checkpt_path):
        """Save model's current parameters at checkpt_path.

        Args:
            checkpt_path (string): path where to save parameters.
        """
        pass
        raise NotImplementedError()

    def recover_or_init(self, init_path):
        """Recover or initialize model based on init_path.

        If init_path has appropriate model parameters, load them; otherwise,
        initialize parameters randomly.
        Args:
            init_path (string): path from where to load parameters.
        """
        pass  #nothing to do here for the moment ... move along
        #raise NotImplementedError()


def one_hot(x, code_size, device):
    """
    Converts input tensor x to One Hot encoding
    code_size is vector size of the one_hot encoded input value
    Returns a tensor with one more dimension of code_size at the end of the input vector x
    """
    #TODO make this with sparse vectors instead
    # print("size = ", code_size ,x.shape)
    out = torch.zeros(x.shape + torch.Size([code_size])).to(device)
    dim = len(x.shape)
    index = x.view(x.shape + torch.Size([1])).long()
    return out.scatter_(dim, index, 1.)
