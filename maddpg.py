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
        self.create_input_placeholders()
        # INITIALIZE ACTOR & CRITIC MODELS
        self.agents = [Actor(self.sess, self.inputs, i, **self.actor_params)
                       for i in [1, 2, 3]]
        self.critic = Critic(self.sess, self.inputs, **self.critic_params)
        # INITIALIZE EXPLORATION MODEL
        self.noise_params = {k: np.fromstring(v, sep=",", dtype="f")
                             for k, v in self.noise_params.items()}
        self.noise = [Noise(**self.noise_params) for _ in range(3)]
        # INITIALIZE REPLAY BUFFER
        self.memory = Memory(self.memory_size)
        # AVERAGE AGENT POLICIES
        avg_pi = [tf.reduce_mean(i, axis=0) for i in zip(*[x.pi.net_params for x in self.agents])]
        self.avg_op = [tf.assign(i, j) for x in self.agents for i, j in zip(x.pi.net_params, avg_pi)]

    def create_input_placeholders(self):
        self.inputs = {}
        ph = lambda s, n: tf.placeholder(tf.float32, shape=s, name=n)
        with tf.name_scope("inputs"):
            self.inputs["g"] = ph((None, 6), "goal")
            self.inputs["d"] = ph((None, 8), "done")
            self.inputs["p"] = ph((None, 1), "pred_q")
            self.inputs["r"] = ph((None, 8), "reward")
            self.inputs["x"] = ph((None, 12), "state")
            self.inputs["o1"] = ph((None, 8), "obs1")
            self.inputs["o2"] = ph((None, 8), "obs2")
            self.inputs["o3"] = ph((None, 8), "obs3")
            self.inputs["a1"] = ph((None, 2), "act1")
            self.inputs["a2"] = ph((None, 2), "act2")
            self.inputs["a3"] = ph((None, 2), "act3")
            self.inputs["dqdu"] = ph((None, 2), "dqdu")


    def step(self, obs, goal, state=None, explore=True):
        q = 0.
        sh = (1, -1)
        obs = {i: j.reshape(sh) for i, j in obs.items()}
        goal = goal.reshape(sh)
        state = state.reshape(sh)
        if explore:
            u = [x.predict(obs[i], goal) + self.noise[i]() 
                 for i, x in enumerate(self.agents)]
            # print(u)
        else:
            u = [x.predict_target(obs[i], goal) for i, x in enumerate(self.agents)]
        if state is not None:
            q = self.critic.predict_target(state, goal, *u)
        u = [x.reshape((1, -1)) for x in u]
        return [self.scale_u(x) for x in u], u, float(q)

    def remember(self, experience):
        self.memory.add(experience)

    def train(self):
        # check if the memory contains enough experiences
        if self.memory.size < 2*self.b_size:
            return
        x, g, o1, o2, o3, a1, a2, a3, r, d, x2, ag, o21, o22, o23 = self.get_batch()
        # print(a1)
        # HER TRANSACTIONS
       
        # her_idxs = np.where(np.random.random(self.b_size).reshape((-1, 1)) < 0.80)
        # g[her_idxs] = ag[her_idxs]
        # r[her_idxs] = 2.
        # d[her_idxs] = True
        obs = [o1, o2, o3]
        n_o = [o21, o22, o23]
        n_u = [j.predict_target(n_o[i], g) for i, j in enumerate(self.agents)]
        n_q = self.critic.predict_target(x2, g, *n_u)
        t_q = r + self.gamma*n_q*(1 - d)
        self.critic.train(x, g, a1, a2, a3, t_q)
        grad = self.critic.get_action_grads(x, g, a1, a2, a3)
        [j.train(obs[i], g, grad[i][0]) for i, j in enumerate(self.agents)]
        # self.update_targets()

    def get_batch(self):
        batch = self.memory.sample(self.b_size)
        return [np.vstack([experience[j] for experience in batch]) for j in range(15)]

    def update_targets(self):
        self.critic.update_target()
        [x.update_target() for x in self.agents]
        self.sess.run(self.avg_op)
