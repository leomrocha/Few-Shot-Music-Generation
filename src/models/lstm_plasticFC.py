import numpy as np
import tensorflow as tf

from models.tf_model import TFModel
from models.base_model import convert_tokens_to_input_and_target


class LSTMPlasticFC(TFModel):
    """LSTM language model

    Trained on songs from the meta-training set. During evaluation,
    ignore each episode's support set and evaluate only on query set.
    """

    def __init__(self, config):
        super(LSTMPlasticFC, self).__init__(config)

    def _define_placedholders(self):
        # Add start word that starts every song
        # Adding start word increases the size of vocabulary by 1
        self._start_word = self._config['input_size']
        self._input_size = self._config['input_size'] + 1

        self._time_steps = self._config['max_len']
        self._embd_size = self._config['embedding_size']
        self._hidden_size = self._config['hidden_size']
        self._n_layers = self._config['n_layers']

        self._plastic_size = self._config['plastic_size']
        self._n_plastic_layers = self._config['n_plastic_layers']
        self._eta = self._config['plastic_eta']
        # self._rule = self._config['plastic_rule']

        self._lr = self._config['lr']
        self._max_grad_norm = self._config['max_grad_norm']

        self._batch_size = tf.placeholder(tf.int32, shape=())
        self._seq_length = tf.placeholder(tf.int32, [None])
        self._words = tf.placeholder(
            tf.int32, [None, self._time_steps])
        self._target = tf.placeholder(
            tf.int32, [None, self._time_steps])

        # self._hebb_1 = tf.placeholder(tf.float32, [self._plastic_size, self._plastic_size])
        # self._hebb_2 = tf.placeholder(tf.float32, [self._plastic_size, self._plastic_size])

        _y_batch_size = 20  # tf.shape(self._batch_size)[0]
        self._x_prev_1 = tf.get_variable("_x_prev_1", [_y_batch_size * self._time_steps, self._plastic_size], dtype=tf.float32,
                                         initializer=tf.zeros_initializer)
        self._x_prev_2 = tf.get_variable("_x_prev_2", [_y_batch_size * self._time_steps, self._plastic_size], dtype=tf.float32,
                                         initializer=tf.zeros_initializer)

    def _build_graph(self):
        embedding = tf.get_variable(
            'embedding', [self._input_size, self._embd_size])
        inputs = tf.nn.embedding_lookup(embedding, self._words)
        inputs = tf.unstack(inputs, axis=1)

        def make_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                self._hidden_size, forget_bias=1., state_is_tuple=True)

        self._cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell() for _ in range(self._n_layers)])
        self._initial_state = self._cell.zero_state(
            self._batch_size, dtype=tf.float32)
        outputs, state = tf.nn.static_rnn(
            self._cell, inputs, initial_state=self._initial_state,
            sequence_length=self._seq_length)
        self._state = state

        #dynamic shape for output values
        _y_batch_size = 20  # tf.shape(self._batch_size)[0]
        # print("batch size shape = ", _y_batch_size)

        output_1 = tf.concat(outputs, 1)
        in_hebb_1 = tf.reshape(output_1, [-1, self._hidden_size])
        # Hebbian Plastic learning in fully connected networks
        plastic_w_1 = tf.get_variable("plastic_w_1", [self._plastic_size, self._plastic_size])
        plastic_alpha_1 = tf.get_variable("plastic_alpha_1", [self._plastic_size, self._plastic_size])
        hebb_1 = tf.get_variable("hebb_1", [self._plastic_size, self._plastic_size])
        y_out_1 = tf.get_variable("y_out_1", [_y_batch_size * self._time_steps, self._plastic_size])

        plastic_w_2 = tf.get_variable("plastic_w_2", [self._plastic_size, self._plastic_size])
        plastic_alpha_2 = tf.get_variable("plastic_alpha_2", [self._plastic_size, self._plastic_size])
        hebb_2 = tf.get_variable("hebb_2", [self._plastic_size, self._plastic_size])
        y_out_2 = tf.get_variable("y_out_2", [_y_batch_size * self._time_steps, self._plastic_size])

        #update hebbian

        in_flow_1 = tf.matmul(in_hebb_1, plastic_w_1)
        # print("in_flow_1.shape, in_hebb_1.shape = ", in_flow_1.get_shape().as_list(), in_hebb_1.get_shape().as_list())
        hebb_factor_1 = tf.multiply(plastic_alpha_1, hebb_1)
        # print("plastic_alpha_1, hebb_factor_1.shape = ", plastic_alpha_1.get_shape().as_list(), hebb_factor_1.get_shape().as_list())
        hebb_flow_1 = tf.matmul(in_hebb_1, hebb_factor_1)
        # print("hebb_flow_1, in_hebb_1.shape = ", hebb_flow_1.get_shape().as_list(), in_hebb_1.get_shape().as_list())
        hebb_sum_1 = in_flow_1 + hebb_flow_1
        # print("hebb_sum_1.shape = ", hebb_sum_1.get_shape().as_list())
        res_hebb_1 = tf.tanh(hebb_sum_1)
        # print("res_hebb_1.shape = ", res_hebb_1.get_shape().as_list())
        tf.assign(y_out_1, res_hebb_1)
        # print("hebb_1.shape update = ", hebb_1.get_shape().as_list(), hebb_1.get_shape().as_list())
        # self._hebb_1 = hebb_1

        # tf.assign(y_out_1,
        #           tf.tanh(tf.matmul(plastic_w_1, in_hebb_1) + tf.matmul(tf.matmul(plastic_alpha_1, hebb_1), in_hebb_1)))
        # Basic Hebbian update rule
        # hebb_ij(t+1) = eta * x_i(t-1) * x_j(t) + (1-eta) * Hebb_ij(t)
        # check tensorflow einsum for outer product here instead of  *
        # print("hebb update shapes = ", self._x_prev_1.get_shape().as_list(), y_out_1.get_shape().as_list(), self._eta, hebb_1.get_shape().as_list())
        # y_outer_1 = tf.einsum('i,j->ij', self._x_prev_1, tf.transpose(y_out_1))
        # y_outer_1 = tf.tensordot(self._x_prev_1, tf.transpose(y_out_1), axes=0)
        y_outer_1 = tf.matmul(tf.transpose(self._x_prev_1), y_out_1)
        print(y_outer_1.get_shape().as_list())
        tf.assign(hebb_1, self._eta * y_outer_1 + (1.-self._eta) * hebb_1)
        #save value for next iteration
        self._x_prev_1 = y_out_1
        # assign second fully connected layer
        tf.assign(y_out_2,
                  tf.tanh(tf.matmul(y_out_1, plastic_w_2) + tf.matmul(y_out_1, tf.multiply(plastic_alpha_2, hebb_2))))
        tf.assign(hebb_2, self._eta * tf.matmul(tf.transpose(self._x_prev_2), y_out_2) + (1.-self._eta) * hebb_2)
        # save value for next iteration
        self._x_prev_2 = y_out_2

        self._output = y_out_2
        # print("y_out_2.shape = ", y_out_2.get_shape().as_list())
        #END hebbian layers

        # softmax_w = tf.get_variable('softmax_w', [self._hidden_size, self._input_size])
        softmax_w = tf.get_variable('softmax_w', [self._plastic_size, self._input_size])
        softmax_b = tf.get_variable('softmax_b', [self._input_size])
        # print("softmax W and B .shapes = ", softmax_w.get_shape().as_list(), softmax_b.get_shape().as_list())
        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.nn.xw_plus_b(self._output, softmax_w, softmax_b)
        # print("logits.shape = ", logits.get_shape().as_list())
        logits = tf.reshape(
            logits, [self._batch_size, self._time_steps, self._input_size])
        self._logits = logits
        self._prob = tf.nn.softmax(self._logits)

        # print("last shapes before log loss = ", self._logits.get_shape().as_list(), self._prob.get_shape().as_list(), self._target.get_shape().as_list(),)
        self._avg_neg_log = tf.contrib.seq2seq.sequence_loss(
            logits,
            self._target,
            tf.ones([self._batch_size, self._time_steps], dtype=tf.float32),
            average_across_timesteps=True,
            average_across_batch=True)

        lr = tf.train.exponential_decay(
            self._lr,
            self._global_step,
            self._config['n_decay'], 0.5, staircase=False
        )
        optimizer = tf.train.AdamOptimizer(lr)
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._avg_neg_log,
                                                       self.get_vars()),
                                          self._max_grad_norm)
        self._train_op = optimizer.apply_gradients(zip(grads, self.get_vars()),
                                                   self._global_step)

    def train(self, episode):
        """Concatenate query and support sets to train."""
        X, Y = convert_tokens_to_input_and_target(
            episode.support, self._start_word)
        X2, Y2 = convert_tokens_to_input_and_target(
            episode.query, self._start_word)
        X = np.concatenate([X, X2])[:20, :]  # FIXME dimensions fixed by hand just to see if the idea works
        Y = np.concatenate([Y, Y2])[:20, :]  # FIXME dimensions fixed by hand just to see if the idea works
        # print("train shapes X,Y = ", X.shape, Y.shape)
        # self._Xshape = X.shape

        feed_dict = {}
        feed_dict[self._words] = X
        feed_dict[self._target] = Y
        feed_dict[self._batch_size] = np.shape(X)[0]  #FIXME this should be of the correct shape
        feed_dict[self._seq_length] = [np.shape(X)[1]] * np.shape(X)[0]  #FIXME this should be of the correct shape

        _, loss = self._sess.run([self._train_op, self._avg_neg_log],
                                 feed_dict=feed_dict)
        if self._summary_writer:
            summary = tf.Summary(value=[
                tf.Summary.Value(tag='Train/loss',
                                 simple_value=loss)])
            self._summary_writer.add_summary(summary, self._train_calls)
            self._train_calls += 1

        return loss

    def eval(self, episode):
        """Ignore support set and evaluate only on query set."""
        X, Y = convert_tokens_to_input_and_target(
            episode.query, self._start_word)
        # print("eval shapes X,Y = ", X.shape, Y.shape)
        # self._Xshape = X.shape

        feed_dict = {}
        feed_dict[self._words] = X
        feed_dict[self._target] = Y
        feed_dict[self._batch_size] = np.shape(X)[0]
        feed_dict[self._seq_length] = [np.shape(X)[1]] * np.shape(X)[0]
        avg_neg_log = self._sess.run(self._avg_neg_log, feed_dict=feed_dict)
        if self._summary_writer is not None:
            summary = tf.Summary(value=[
                tf.Summary.Value(tag='Eval/Avg_NLL',
                                 simple_value=avg_neg_log)])
            self._summary_writer.add_summary(summary, self._eval_calls)
            self._eval_calls += 1

        return avg_neg_log

    def sample(self, support_set, num):
        """Ignore support set for sampling."""
        pred_words = []
        word = self._start_word

        state = self._sess.run(self._cell.zero_state(1, tf.float32))
        x = np.zeros((1, self._time_steps))
        for i in range(num):
            x[0, 0] = word
            feed_dict = {}
            feed_dict[self._words] = x
            feed_dict[self._batch_size] = 1
            feed_dict[self._seq_length] = [1]
            feed_dict[self._initial_state] = state

            probs, state = self._sess.run([self._prob, self._state],
                                          feed_dict=feed_dict)
            p = probs[0][0]
            word = np.argmax(p)
            pred_words.append(word)

        return pred_words
