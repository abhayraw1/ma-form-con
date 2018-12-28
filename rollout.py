import time
import numpy as np
import tensorflow as tf
from util import info, log


class RolloutGenerator:
    """
    Class for generating a rollout of trajectory by the agent
    args:
    env: gym env
    agent: agent for performing rollout
    config: rollout configuration
    checkpoint(opt): perform rollout from a saved policy
    """

    def __init__(self, env, agent, config: dict, _eval=False, summarize=False):
        self.env = env
        self.agent = agent
        self.eval = _eval
        self.best_score = 0.
        self.__dict__.update(config)
        self.saver = tf.train.Saver()
        self.p_ckpt = "__checkpoints/{}_{}"
        self.name = ["*TRAINING*", "EVALUATION"][int(_eval)]
        if "periodic_ckpt" not in self.__dict__:
            self.periodic_ckpt = False
        if "save_best" not in self.__dict__:
            self.save_best = False
        self.reset()
        metrics = ["", "EPISODE.", "REWARD.", "TIMESTEPS.", "AVG_Q.", ""]
        self.logstr = "||".join(i.replace(".", ": {}") for i in metrics)
        info.out("INITIALIZED {} ROLLOUT GENERATOR".format(self.name))

    def reset(self):
        self.q_total = 0.
        self.r_total = 0.
        self.t_steps = 0
        self.episode = 0
        self.successes = 0

    def generate_rollout(self):
        t = 0
        done = False
        episodic_q = 0.
        episodic_r = 0.
        obs, xs, xh = self.env.reset()
        x = np.hstack([xs, xh])
        g = self.env.goal
        while not done and t < self.env.max_episode_steps:
            a, u, q = self.agent.step(obs, g, x, (not self.eval))
            obs2, xs2, xh2, r, done, info = self.env.step(dict(enumerate(a)))
            x2 = np.hstack([xs2, xh2])
            self.agent.remember([x, g, *obs.values(), *a, r, done, x2, xs2, *obs2.values()])
            x = x2

            # Render if required
            if "render" in self.__dict__ and self.render:
                self.env.render()

            # Update stats
            t += 1
            episodic_r += float(r)
            episodic_q += float(q)

            # Train agent if required
            if not self.eval:
                [self.agent.train() for _ in range(self.train_cycles_per_ts)]
            else:
                if "step_sleep" in self.__dict__:
                    time.sleep(self.step_sleep)
        self.episode += 1
        self.update_stats(episodic_q, episodic_r, t)
        self.successes += 1 if done else 0
        log.out(self.logstr.format(self.episode, episodic_r, t, episodic_q/t))
        self.create_checkpoint()

    def create_checkpoint(self):
        if self.periodic_ckpt and self.episode % self.periodic_ckpt == 0:
            info.out("Creating periodic checkpoint")
            self.saver.save(self.agent.sess,
                            self.p_ckpt.format("P", self.episode))
        if self.eval and self.save_best and self.mean_er > self.best_score:
            info.out("New best score: {}".format(self.mean_er))
            self.best_score = self.mean_er
            self.saver.save(self.agent.sess,
                            self.p_ckpt.format("B", self.episode))

    def update_stats(self, eps_q, eps_r, t):
        self.q_total += eps_q
        self.r_total += eps_r
        self.t_steps += t
        self.mean_eq = self.q_total/self.episode
        self.mean_er = self.r_total/self.episode


    def done(self):
        done = self.n_episodes <= self.episode
        if done and self.eval:
            print("\n\n")
        return done

    # def summarize(self):
    #     if self.summarizer is None:
    #         return
    #     summarizer.value.add(tag="{}/")
