import os
from torch import nn
import pprint
from models.base_model import BaseModel
import numpy as np

PP = pprint.PrettyPrinter(depth=6)


class PyTorchModel(BaseModel):
    """docstring for PyTorchModel.BaseModel"""
    def __init__(self, config):
        super(PyTorchModel,self).__init__(config)
        self._start_word = self._config['input_size']
        self._input_size = self._config['input_size'] + 1

        self._time_steps = self._config['max_len']
        self._embd_size = self._config['embedding_size']
        self._lr = self._config['lr']
        self._max_grad_norm = self._config['max_grad_norm']

        self.criterion = nn.MSELoss() #this loss does not have backward
        #criterion = nn.CrossEntropyLoss() # FIXME this loss does not work with the current setup

    def _to_one_hot(self, data):
        pass# TODO

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
        # create embedding
        X = self._to_one_hot(X)
        Y = self._to_one_hot(Y)
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
        return loss

    def eval(self, episode):
        """Ignore support set and evaluate only on query set."""
        print(episode.query.shape)
        X, Y = convert_tokens_to_input_and_target(episode.query, self._start_word)
        X = torch.from_numpy(X).to(self.device)
        Y = torch.from_numpy(Y).to(self.device)
        # create embedding
        X = self._to_one_hot(X)
        Y = self._to_one_hot(Y)
        print(X.shape, Y.shape)
        out = self.model(X)
        print(out.shape)
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


def convert_tokens_to_input_and_target(token_array, start_word=None):
    """Convert token_array to input and target to use for model for
    sequence generation.

    If start_word is given, add to start of each sequence of tokens.
    Input is token_array without last item; Target is token_array without first item.

    Arguments:
        token_array (numpy int array): tokens array of size [B,S,N] where
            B is batch_size, S is number of songs, N is size of the song
        start_word (int): token to use for start word

    Returns:
        arrays of dimensions:
        [sec_len, batch_size, input_size]
    """
    print("token_array.shape = ", token_array.shape)
    X = np.transpose(token_array, (1,0,2))

    if start_word is None:
        Y = np.copy(X[:, :, 1:])
        X_new = X[:, :, :-1]
    else:
        Y = np.copy(X)
        start_word_column = np.full(
            shape=[np.shape(X)[0],np.shape(X)[1], 1], fill_value=start_word)
        X_new = np.concatenate([start_word_column, X[:, :, :-1]], axis=2)

    return X_new, Y
