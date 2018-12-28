import tensorflow as tf
from tensorflow import GraphKeys
from tensorflow.layers import dense
from tensorflow.initializers import truncated_normal as TN
TVARS = GraphKeys.TRAINABLE_VARIABLES


class FCNN:
    def __init__(self, _input, op_dim, n_layers, n_units, activation,
                 op_act=None, name="FCNN", w_init=None, from_ckpt=False):
        print("CREATING NETWORK NAMED {}".format(name))
        self._input = _input
        self.op_dim = op_dim
        self.n_layers = n_layers
        self.n_units = n_units
        self.activation = activation
        self.scope = name
        self.op_act = op_act
        self.w_init = w_init if w_init is not None else TN(stddev=1e-1)
        self.make()

    def make(self):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            _input = self._input
            for i in range(0, self.n_layers-1):
                op = dense(_input, self.n_units,
                           kernel_initializer=self.w_init,
                           name="layer_{}".format(i))
                _input = self.activation(op)
            op = dense(_input, self.op_dim,
                       kernel_initializer=self.w_init,
                       name="layer_{}".format(i+1))
            self.nn = op if self.op_act is None else self.op_act(op)
            self.net_params = tf.get_collection(TVARS, scope=self.scope)
        for i in self.net_params:
            tf.summary.histogram(i.name.replace(":", "_"), i)

    def __call__(self, sess, inputs):
        return sess.run(self.nn, inputs)
