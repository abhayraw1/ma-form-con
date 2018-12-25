import tensorflow as tf

from tensorflow import square as sq
from tensorflow import multiply as mul
from tensorflow import reduce_mean as rmean

from tensorflow.train import AdamOptimizer as Adam

from FCNN import FCNN


class Critic:
    def __init__(self, sess, input_t, **params):
        self.session = sess
        self.input_t = input_t
        self.__dict__.update(params)
        self.generate_networks()
        self.define_operations()

    def generate_networks(self):
        q_input = tf.concat([self.x, self.g, self.a1, self.a2, self.a3], axis=1)
        # MAIN CRITIC NETWORK
        self.q = FCNN(q_input, 1, self.n_layers, self.n_units,
                      tf.nn.relu, name="q")
        # TARGET CRITIC NETWORK
        self.Q = FCNN(q_input, 1, self.n_layers, self.n_units,
                      tf.nn.relu, name="t_q")

    def define_operations(self):
        with tf.name_scope("critic_ops"):
            # LOSS
            loss = tf.sqrt(rmean(sq(self.p - self.q.nn)))
            # MINIMIZE LOSS OP
            self.minimize = Adam(self.lr, name="q_adam")\
                .minimize(loss, var_list=self.q.net_params)
            # ACTION GRADIENTS
            self.a_grads1 = tf.gradients(self.q.nn, self.a1, name="dq_da1")
            self.a_grads2 = tf.gradients(self.q.nn, self.a2, name="dq_da2")
            self.a_grads3 = tf.gradients(self.q.nn, self.a3, name="dq_da3")
            self.a_grad_ops = [self.a_grads1, self.a_grads2, self.a_grads3]
            # UPDATE TARGET OP
            net_param_pairs = zip(self.q.net_params, self.Q.net_params)
            with tf.name_scope("update_target_q"):
                self.updt_Q = [j.assign(mul(self.tau, i)+mul((1-self.tau), j))
                               for i, j in net_param_pairs]

    def predict(self, x, g, a1, a2, a3):
        feed_vals = {self.x: x, self.g: g, self.a1: a1,
                     self.a2: a2, self.a3: a3}
        return self.session.run(self.q.nn, feed_dict=feed_vals)

    def predict_target(self, x, g, a1, a2, a3):
        feed_vals = {self.x: x, self.g: g, self.a1: a1,
                     self.a2: a2, self.a3: a3}
        return self.session.run(self.Q.nn, feed_dict=feed_vals)

    def train(self, x, g, a1, a2, a3, p):
        feed_vals = {self.x: x, self.g: g, self.a1: a1,
                     self.a2: a2, self.a3: a3, self.p: p}
        return self.session.run([self.q.nn, self.minimize], feed_dict=feed_vals)

    def get_action_grads(self, x, g, a1, a2, a3):
        feed_vals = {self.x: x, self.g: g, self.a1: a1,
                     self.a2: a2, self.a3: a3}
        return self.session.run(self.a_grad_ops, feed_dict=feed_vals)

    def update_target(self):
        self.session.run(self.updt_Q)
