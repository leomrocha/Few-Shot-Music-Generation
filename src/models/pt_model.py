import os
import torch
from torch import nn
import pprint
from models.base_model import BaseModel, convert_tokens_to_input_and_target
import numpy as np

PP = pprint.PrettyPrinter(depth=6)


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
        # self.criterion = nn.MSELoss() #this loss does not have backward
        # self.criterion = nn.CrossEntropyLoss() # FIXME this loss does not work with the current setup
        self.criterion = nn.BCELoss()

    def _encode(self, data):
        return one_hot(data, self._input_size, self.device)
        # return self.encoder(data)

    def train(self, episode):
        """Train model on episode.

        Args:
            episode: Episode object containing support and query set.
        """
        # print("training PyTorch Model")
        # format input data
        X, Y = convert_tokens_to_input_and_target(
            episode.support, self._start_word)
        X2, Y2 = convert_tokens_to_input_and_target(
            episode.query, self._start_word)
        X = np.concatenate([X, X2])
        X = torch.from_numpy(X).to(self.device)
        Y = np.concatenate([Y, Y2])
        Y = torch.from_numpy(Y).to(self.device)
        # print(" 3 grad ? ", X.requires_grad, Y.requires_grad)
        # X.grad = None
        # Y.grad = None
        # print("4 grad ? ", X.requires_grad, Y.requires_grad)
        # create embedding
        # X = self._encode(X)
        Y = self._encode(Y)
        # print("5 grad ? ", X.requires_grad, Y.requires_grad)
        # train
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr)
        self.model.hidden = self.model.init_hidden()  # reset state to avoid interference between different episodes
        self.model.zero_grad()
        # out = self.model(X, future=self._time_steps)
        out = self.model(X)
        # print(out.shape)
        loss = self.criterion(out, Y)
        # print(type(loss))
        loss.backward()
        optimizer.step()
        try:
            X = Y = X2 = Y2 = None
            torch.cuda.empty_cache()
        except:
            print("error emptying memory")
            pass
        return loss

    def eval(self, episode):
        """Ignore support set and evaluate only on query set."""
        # print("episode query shape = ",episode.query.shape)
        X, Y = convert_tokens_to_input_and_target(episode.query, self._start_word)
        X = torch.from_numpy(X).to(self.device)
        Y = torch.from_numpy(Y).to(self.device)
        # print("1- grad ? ", X.requires_grad, Y.requires_grad)
        # X.grad = None
        # Y.grad = None
        # print("2- grad ? ", X.requires_grad, Y.requires_grad)
        # create embedding
        # X = self._encode(X)
        Y = self._encode(Y)
        # print("eval 3- grad ? ", X.requires_grad, Y.requires_grad)
        # print("X,Y shapes = ", X.shape, Y.shape)

        out = self.model(X)
        # print("eval 4- grad ? ", out.requires_grad) # ->grad =True
        # print("out.shape = ", out.shape)
        # print("tensor types = ", type(out), type(Y))
        # FIXME shape here is wrong -> is the issue with the shape of the embedding ...
        loss = self.criterion(out.view(-1, out.shape[-1]), Y.view(-1, Y.shape[-1]))
        # print("eval 5- grad ? ", loss.requires_grad) # ->grad =True
        try:
            X = Y = out = None
            torch.cuda.empty_cache()
        except:
            print("error emptying memory")
            pass
        loss = loss.data.cpu().numpy()
        # print("Evaluation loss = ", loss)
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
        #raise NotImplementedError()

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
    # TODO make this with sparse vectors instead
    # print("size = ", code_size ,x.shape)
    out = torch.zeros(x.shape + torch.Size([code_size])).to(device)
    # print("one_hot grad ? ", out.requires_grad)
    dim = len(x.shape)
    index = x.view(x.shape + torch.Size([1])).long()
    return out.scatter_(dim, index, 1.)
