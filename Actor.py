import tensorflow as tf

from tensorflow import multiply as mul

from tensorflow.train import AdamOptimizer as Adam
from tensorflow.initializers import truncated_normal as TN

from FCNN import FCNN


class Actor:
    def __init__(self, sess, input_t, **params):
        self.session = sess
        self.load_from_ckpt = False
        self.__dict__.update(params)
        self.__dict__.update(input_t)
        self.generate_networks()
        self.define_operations()

    def generate_networks(self, load_from_ckpt=False):
        # MAIN ACTOR NETWORK
        self.pi = FCNN(self.x, self.u.shape[-1], self.n_layers, self.n_units,
                       tf.nn.relu, tf.nn.tanh, name="pi",
                       w_init=TN(stddev=1e-1), from_ckpt=self.load_from_ckpt)
        # TARGET ACTOR NETWORK
        self.PI = FCNN(self.x, self.u.shape[-1], self.n_layers, self.n_units,
                       tf.nn.relu, tf.nn.tanh, name="t_pi",
                       from_ckpt=self.load_from_ckpt)

    def define_operations(self):
        with tf.name_scope("actor_ops"):
            # GRADIENT OF ACTIONS WRT ACTOR PARAMS TIMES NEGATIVE GRADIENT OF
            # VALUE FUNCTION WRT ACTIONS
            grads = tf.gradients(self.pi.nn, self.pi.net_params, -self.g)
            # APPLY GRADIENTS TO ACTOR NETWORK
            self.optimize = Adam(self.lr, name="pi_adam")\
                .apply_gradients(zip(grads, self.pi.net_params))
            # UPDATE TARGET OP
            net_param_pairs = zip(self.pi.net_params, self.PI.net_params)
            with tf.name_scope("update_target_pi"):
                self.updt_PI = [j.assign(mul(self.tau, i)+mul((1-self.tau), j))
                                for i, j in net_param_pairs]

    def predict(self, x):
        return self.session.run(self.pi.nn,
                                feed_dict={self.x: x})

    def predict_target(self, x):
        return self.session.run(self.PI.nn,
                                feed_dict={self.x: x})

    def train(self, x, g):
        return self.session.run(self.optimize,
                                feed_dict={self.x: x, self.g: g})

    def update_target(self):
        self.session.run(self.updt_PI)
