import tensorflow as tf

from tensorflow import square as sq
from tensorflow import multiply as mul
from tensorflow import reduce_mean as rmean

from tensorflow.train import AdamOptimizer as Adam

from FCNN import LSTM_FCNN


class Critic:
    def __init__(self, sess, input_t, **params):
        self.session = sess
        self.ip = input_t
        self.__dict__.update(params)
        self.generate_networks()
        self.define_operations()

    def generate_networks(self):
        q_input = tf.concat([self.ip, self.a1, self.a2, self.a3], axis=2)
        print(q_input)
        # MAIN CRITIC NETWORK
        self.q = LSTM_FCNN(q_input, 128, 1, self.n_layers, self.n_units,
                      tf.nn.relu, name="q")
        # TARGET CRITIC NETWORK
        self.Q = LSTM_FCNN(q_input, 128, 1, self.n_layers, self.n_units,
                      tf.nn.relu, name="t_q")

    def define_operations(self):
        # with tf.name_scope("critic_ops"):
            # LOSS
        loss = tf.sqrt(rmean(sq(self.pr - self.q.nn)))
        # MINIMIZE LOSS OP
        self.minimize = Adam(self.lr, name="q_adam")\
            .minimize(loss, var_list=self.q.net_params)
        # ACTION GRADIENTS
        a_grads1 = tf.gradients(self.q.nn, self.a1, name="dq_da1")
        a_grads2 = tf.gradients(self.q.nn, self.a2, name="dq_da2")
        a_grads3 = tf.gradients(self.q.nn, self.a3, name="dq_da3")
        self.a_grad_ops = [a_grads1, a_grads2, a_grads3]
        # UPDATE TARGET OP
        net_param_pairs = zip(self.q.net_params, self.Q.net_params)
        with tf.name_scope("update_target_q"):
            self.updt_Q = [j.assign(mul(self.tau, i)+mul((1-self.tau), j))
                           for i, j in net_param_pairs]

    def predict(self, ip, a1, a2, a3):
        feed_vals = {self.ip: ip, self.a1: a1, self.a2: a2, self.a3: a3}
        return self.session.run(self.q.nn, feed_dict=feed_vals)

    def predict_target(self, ip, a1, a2, a3):
        feed_vals = {self.ip: ip, self.a1: a1, self.a2: a2, self.a3: a3}
        return self.session.run(self.Q.nn, feed_dict=feed_vals)

    def train(self, ip, a1, a2, a3, p):
        feed_vals = {self.ip: ip, self.a1: a1, self.a2: a2, self.a3: a3,
                     self.pr: p}
        return self.session.run(self.minimize, feed_dict=feed_vals)

    def get_action_grads(self, ip, a1, a2, a3):
        feed_vals = {self.ip: ip, self.a1: a1, self.a2: a2, self.a3: a3}
        return self.session.run(self.a_grad_ops, feed_dict=feed_vals)

    def update_target(self):
        self.session.run(self.updt_Q)
