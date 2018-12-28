import tensorflow as tf

from tensorflow import multiply as mul

from tensorflow.train import AdamOptimizer as Adam
from tensorflow.initializers import truncated_normal as TN

from FCNN import FCNN


class Actor:
    def __init__(self, sess, input_t, _id, **params):
        self.session = sess
        self._id = _id
        self.__dict__.update(params)
        self.__dict__.update(input_t)
        self.obs = input_t["o{}".format(_id)]
        self.u = input_t["a{}".format(_id)]
        self.generate_networks()
        self.define_operations()

    def generate_networks(self, load_from_ckpt=False):
        pi_input = tf.concat([self.obs, self.g], axis=1)
        # MAIN ACTOR NETWORK
        self.pi = FCNN(pi_input, self.u.shape[-1], self.n_layers, self.n_units,
                       tf.nn.relu, tf.nn.tanh, name="pi_{}".format(self._id),
                       w_init=TN(stddev=1e-1))
        # TARGET ACTOR NETWORK
        self.PI = FCNN(pi_input, self.u.shape[-1], self.n_layers, self.n_units,
                       tf.nn.relu, tf.nn.tanh, name="t_pi_{}".format(self._id))

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

    def predict(self, obs, g):
        return self.session.run(self.pi.nn,
                                feed_dict={self.obs: obs, self.g: g})

    def predict_target(self, obs, g):
        return self.session.run(self.PI.nn,
                                feed_dict={self.obs: obs, self.g: g})

    def train(self, obs, g, dqdu):
        return self.session.run(self.optimize,
                                feed_dict={self.obs: obs, self.g: g,
                                           self.dqdu: dqdu})

    def update_target(self):
        self.session.run(self.updt_PI)
