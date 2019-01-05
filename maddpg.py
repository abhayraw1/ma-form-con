import numpy as np
import tensorflow as tf


from Actor import Actor
from noise import Noise
from Critic import Critic
from memory import Memory

from util import error

class MADDPG:

    def __init__(self, sess, scale_u, params):
        self.sess = sess
        self.scale_u = scale_u
        self.__dict__.update(params)
        # CREATE INPUT PLACEHOLDERS
        inputs = self.create_input_placeholders()
        # INITIALIZE ACTOR & CRITIC MODELS
        self.policy = Actor(self.sess, inputs["obs"], **self.actor_params)
        self.critic = Critic(self.sess, inputs["state"], **self.critic_params)
        # INITIALIZE EXPLORATION MODEL
        self.noise_params = {k: np.fromstring(v, sep=",", dtype="f")
                             for k, v in self.noise_params.items()}
        self.noise = [Noise(**self.noise_params) for _ in range(3)]
        # INITIALIZE REPLAY BUFFER
        self.memory = Memory(self.memory_size)
        
    def create_input_placeholders(self):
        inputs = {}
        ph = lambda s, n: tf.placeholder(tf.float32, shape=s, name=n)
        with tf.name_scope("inputs"):
            inputs["obs"] = ph((None,self.tracelen, 10), "obs")
            inputs["state"] = ph((None,self.tracelen, 14), "state")
            inputs["dqdu"] = ph((None,self.tracelen, 2), "dqdu")
            inputs["a1"] = ph((None,self.tracelen, 2), "a1")
            inputs["a2"] = ph((None,self.tracelen, 2), "a2")
            inputs["a3"] = ph((None,self.tracelen, 2), "a3")
            inputs["pr"] = ph((None,self.tracelen, 1), "pr")
        self.critic_params.update({"a1": inputs["a1"], "a2": inputs["a2"], 
                                   "a3": inputs["a3"], "pr": inputs["pr"]})
        self.actor_params.update({"dqdu": inputs["dqdu"]})
        return inputs

    def step(self, obs, explore=True):
        if explore:
            u = [self.policy.predict(np.array([obs[i]])) + self.noise[i]() 
                 for i in range(3)]
        else:
            u = [self.policy.predict_target([obs[i]]) for i in range(3)]
        u = [x[0, -1] for x in u]
        return [self.scale_u(x) for x in u], u

    def remember(self, experience):
        self.memory.add(experience)

    def train(self):
        # check if the memory contains enough experiences
        if self.memory.size < 5:
            return
        x, o1, o2, o3, a1, a2, a3, r, d, x2, no1, no2, no3 = self.get_batch()

        # HER TRANSACTIONS #####################################################
        # her_idxs = np.where(np.random.random(self.b_size).reshape((-1, 1)) < 0.50)
        # g[her_idxs] = ag[her_idxs]
        # r[her_idxs] = 200.
        # d[her_idxs] = True
        ########################################################################
        na = [self.policy.predict_target(i) for i in [no1, no2, no3]]
        tq = r + self.gamma*self.critic.predict_target(x2, *na)*(1 - d)
        self.critic.train(x, a1, a2, a3, tq)
        grads = [i[0] for i in self.critic.get_action_grads(x, a1, a2, a3)]
        [self.policy.train(i, j) for i, j in zip([o1, o2, o3], grads)]
        # n_u = [self.policy.predict_target(x, g) for x in n_o]
        # n_q = self.critic.predict_target(x2, g, *n_u)
        # t_q = r + self.gamma*n_q*(1 - d)
        # grad = self.critic.get_action_grads(x, g, a1, a2, a3)
        # [self.policy.train(obs[i], g, grad[i][0]) for i in range(3)]
        # self.update_targets()

    def get_batch(self):
        return self.memory.sample(self.b_size, self.tracelen)

    def update_targets(self):
        self.critic.update_target()
        # [x.update_target() for x in self.agents]
        self.policy.update_target()
        # self.sess.run(self.avg_op)
