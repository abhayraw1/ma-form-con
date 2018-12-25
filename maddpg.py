import numpy as np
import tensorflow as tf


from actor import Actor
from noise import Noise
from critic import Critic
from memory import Memory


class DDPG:

    def __init__(self, sess, scale_u, params):
        self.sess = sess
        self.scale_u = scale_u
        self.__dict__.update(params)
        # create placeholders
        if "inputs" not in params.keys():
            self.create_input_placeholders()
        # create actor/critic models
        self.actor = Actor(self.sess, self.inputs, **self.actor_params)
        self.critic = Critic(self.sess, self.inputs, **self.critic_params)
        self.noise_params = {k: np.array(list(map(float, v.split(","))))
                             for k, v in self.noise_params.items()}
        self.noise = Noise(**self.noise_params)
        self.ou_level = np.zeros(self.dimensions["u"])
        self.memory = Memory(self.n_mem_objects,
                             self.memory_size)

    def create_input_placeholders(self):
        self.inputs = {}
        ph = lambda s, n: tf.placeholder(tf.float32, shape=s, name=n)
        with tf.name_scope("inputs"):
            self.inputs["g"] = ph((None, 6), "goal")
            self.inputs["d"] = ph((None, 8), "done")
            self.inputs["r"] = ph((None, 8), "reward")
            self.inputs["x"] = ph((None, 12), "state")
            self.inputs["o1"] = ph((None, 8), "obs1")
            self.inputs["o2"] = ph((None, 8), "obs2")
            self.inputs["o3"] = ph((None, 8), "obs3")
            self.inputs["a1"] = ph((None, 8), "act1")
            self.inputs["a2"] = ph((None, 8), "act2")
            self.inputs["a3"] = ph((None, 8), "act3")


    def step(self, x, explore=True):
        x = x.reshape(-1, self.dimensions["x"])
        if explore:
            u = self.actor.predict(x)
            self.ou_level = self.noise.ornstein_uhlenbeck_level(self.ou_level)
            # print(self.ou_level, u)
            u = u + self.ou_level
            q = self.critic.predict(x, u)
        else:
            u = self.actor.predict_target(x)
            q = self.critic.predict_target(x, u)
        return [self.scale_u(u[0]), u, q[0]]

    def remember(self, experience):
        self.memory.add(experience)

    def train(self):
        # check if the memory contains enough experiences
        if self.memory.size < 3*self.b_size:
            return
        x, g, ag, u, r, nx, ng, t = self.get_batch()
        # for her transitions
        her_idxs = np.where(np.random.random(self.b_size) < 0.80)[0]
        # print("{} of {} selected for HER transitions".
        # format(len(her_idxs), self.b_size))
        g[her_idxs] = ag[her_idxs]
        r[her_idxs] = 1
        t[her_idxs] = 1
        x = np.hstack([x, g])
        nx = np.hstack([nx, ng])
        nu = self.actor.predict_target(nx)
        tq = r + self.gamma*self.critic.predict_target(nx, nu)*(1-t)
        self.critic.train(x, u, tq)
        grad = self.critic.get_action_grads(x, u)
        # print("Grads:\n", g)
        self.actor.train(x, grad)
        self.update_targets()

    def get_batch(self):
        batch = self.memory.sample(self.b_size)
        return [[experience[j] for experience in batch] for j in range(15)]

    def update_targets(self):
        self.critic.update_target()
        self.actor.update_target()
