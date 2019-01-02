import numpy as np
import tensorflow as tf
from tensorflow.summary import FileWriter

from env import FormEnv
from maddpg import MADDPG
from rollout import RolloutGenerator

env = FormEnv()

def scale_action_gen(env, u_min, u_max):
    def scale_action(u):
        u = np.clip(u, u_min, u_max)
        # print("clipped ", u)
        zo = (u - u_min)/(u_max - u_min)
        return zo * (env.action_high - env.action_low) + env.action_low
    return scale_action

sess = tf.Session()
actor_params = {"n_layers":2, "n_units":128, "tau": 0.01, "lr": 1e-4}
critc_params = {"n_layers":3, "n_units":128, "tau": 0.01, "lr": 1e-3}
noise_params = {"delta": "0.5,0.2", "sigma": "0.5,0.7",
                "ou_a":  "0.6,0.6", "ou_mu": "0.5,0.0"}
params = {"actor_params": actor_params, "b_size": 64,
          "critic_params": critc_params, "gamma": 0.99,
          "noise_params": noise_params, "memory_size": 50000}
agent = MADDPG(sess, scale_action_gen(env, -np.ones(2), np.ones(2)), params)

summarizer = FileWriter("__tensorboard/her", sess.graph)
s_summary = tf.Summary()
summary_op = tf.summary.merge_all()

sess.run(tf.global_variables_initializer())
train_rollouts = RolloutGenerator(env, agent, {"render": 1, "n_episodes": 100000, "periodic_ckpt": 50,
                                  "train_cycles_per_ts":10})
eval_rollouts = RolloutGenerator(env, agent, {"render": 1, "n_episodes": 20, "save_best": True}, True)

while not train_rollouts.done():
    train_rollouts.generate_rollout()
    summarizer.add_summary(sess.run(summary_op), train_rollouts.episode)
    summarizer.flush()
    if (train_rollouts.episode) % 20 == 0:
        eval_rollouts.reset()
        while not eval_rollouts.done():
            eval_rollouts.generate_rollout()