import tensorflow as tf

from tensorflow import multiply as mul

from tensorflow.train import AdamOptimizer as Adam
from tensorflow.initializers import truncated_normal as TN

from FCNN import LSTM_FCNN


class Actor:
    def __init__(self, sess, input_t, **params):
        self.session = sess
        self.ip = input_t
        self.__dict__.update(params)
        self.generate_networks()
        self.define_operations()

    def generate_networks(self):
        print(self.ip, self.n_layers, self.n_units)
        # MAIN ACTOR NETWORK
        self.pi = LSTM_FCNN(self.ip, 128, 2, self.n_layers, self.n_units,
                       tf.nn.relu, tf.nn.tanh, name="pi",
                       w_init=TN(stddev=1e-1))
        # TARGET ACTOR NETWORK
        self.PI = LSTM_FCNN(self.ip, 128, 2, self.n_layers, self.n_units,
                       tf.nn.relu, tf.nn.tanh, name="t_pi")

    def define_operations(self):
        with tf.name_scope("actor_ops"):
            # GRADIENT OF ACTIONS WRT ACTOR PARAMS TIMES NEGATIVE GRADIENT OF
            # VALUE FUNCTION WRT ACTIONS
            grads = tf.gradients(self.pi.nn, self.pi.net_params, -self.dqdu)
            # APPLY GRADIENTS TO ACTOR NETWORK
            self.optimize = Adam(self.lr, name="pi_adam")\
                .apply_gradients(zip(grads, self.pi.net_params))
            # UPDATE TARGET OP
            net_param_pairs = zip(self.pi.net_params, self.PI.net_params)
            with tf.name_scope("update_target_pi"):
                self.updt_PI = [j.assign(mul(self.tau, i)+mul((1-self.tau), j))
                                for i, j in net_param_pairs]

    def predict(self, ip):
        return self.pi(self.session, ip)

    def predict_target(self, ip):
        return self.PI(self.session, ip)

    def train(self, ip, dqdu):
        return self.session.run(self.optimize,
                                feed_dict={self.ip: ip, self.dqdu: dqdu})

    def update_target(self):
        self.session.run(self.updt_PI)
