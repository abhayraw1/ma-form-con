import gym
from gym.spaces import Box
from gym.envs.registration import register
from gym.envs.classic_control import rendering

import numpy as np
from numpy.linalg import norm
from numpy.random import random

from PointEnvironment.Pose import Pose
from PointEnvironment.Agent import Agent

class FormEnv(gym.Env):
    cossin = staticmethod(lambda x: np.array([np.cos(x), np.sin(x)]))

    def __init__(self, config=None):
        self.configure_defaults()
        if config is not None:
            self.__dict__.update(config)
        self.goal = None
        self.viewer = None

    def configure_defaults(self):
        self.dt = 1e-2
        self.num_iter = 50
        self.max_episode_steps = 25
        self.step_penalty = 1.0
        self.max_reward = 2.0
        self.action_low = np.array([0.0, -np.pi/4])
        self.action_high = np.array([0.4, np.pi/4])
        self.action_space = Box(self.action_low, self.action_high, dtype="f")
        self.w_limits = np.array([10, 10])
        self.s_limits = np.array([600, 600])
        self.scale = self.s_limits/self.w_limits
        self.scale = self.scale[0]
        self.agent_radius = 0.15  # in meters
        self.agents = [Agent(i) for i in range(3)]

    def init_viewer(self):
        self.viewer = rendering.Viewer(*self.s_limits)
        lx = [rendering.Line((0, pt), (self.s_limits[1], pt)) for pt in
              np.arange(0, self.s_limits[0], self.scale)]
        ly = [rendering.Line((pt, 0), (pt, self.s_limits[0])) for pt in
              np.arange(0, self.s_limits[1], self.scale)]
        [self.viewer.add_geom(i) for i in lx+ly]
        # GOAL MARKER
        circle = rendering.make_circle(radius=0.15*self.scale)
        circle.set_color(0.3, 0.82, 0.215)
        self.goal_tf = rendering.Transform()
        circle.add_attr(self.goal_tf)
        self.viewer.add_geom(circle)
        # AGENT MARKERS
        self.agent_tfs = []
        a_rad_px = self.agent_radius * self.scale
        verx = [a_rad_px*FormEnv.cossin(np.radians(i)) for i in [0, 140, -140]]
        for i in self.agents:
            agent = rendering.FilledPolygon([tuple(j) for j in verx])
            agent.set_color(0.15, 0.235, 0.459)
            agent_tf = rendering.Transform()
            agent.add_attr(agent_tf)
            self.agent_tfs.append(agent_tf)
            self.viewer.add_geom(agent)
        # CENTROID MARKER
        circle = rendering.make_circle(radius=0.05*self.scale)
        circle.set_color(0.9, 0.3, 0.23)
        self.centroid_tf = rendering.Transform()
        circle.add_attr(self.centroid_tf)
        self.viewer.add_geom(circle)

    def get_form_goal(self):
        sides = np.random.random(3)*0.8 + 0.7
        if np.array([np.sum(sides) - 2*x > 0 for x in sides]).all():
            self.goal_sides = sides
            a, b, c = sides
            coordinates = [[0, 0], [a, 0]]
            h = (c**2 - b**2 + a**2)/(2*a)
            coordinates.append([h, (c**2 - h**2)**.5])
            centroid = np.mean(coordinates, axis=0)
            return np.hstack([i - centroid for i in coordinates])
        return self.get_form_goal()

    def sample_pose(self, limits=None):
        if limits is None:
            x, y = random(2)*self.w_limits - self.w_limits/2
        else:
            x, y = random(2)*limits - limits/2
        theta = (random()*2 - 1)*np.pi
        return Pose(x=x, y=y, t=theta)

    def reset(self):
        [a.reset(self.sample_pose(np.array([2, 2]))) for a in self.agents]
        self.goal = self.get_form_goal()
        self.goal_changed = True
        return self.compute_obs()

    def render(self, mode='human'):
        if self.goal is None:
            return None
        if self.viewer is None:
            self.init_viewer()
        for agent, agent_tf in zip(self.agents, self.agent_tfs):
            agent_tf.set_translation(*(agent.pose.tolist()[:-1] +
                                     self.w_limits//2)*self.scale)
            agent_tf.set_rotation(agent.pose.theta)
        centroid = np.mean([a.pose.tolist()[:-1] for a in self.agents], 0)
        self.centroid_tf.set_translation(*(centroid+self.w_limits//2)
                                         * self.scale)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def compute_obs(self):
        obs = {j.id: np.hstack([np.hstack(j.pose.getPoseInFrame(i.pose))
               for i in self.agents if i.id != j.id]) for j in self.agents}
        f_c = np.mean([a.pose.tolist()[:-1] for a in self.agents], axis=0)
        cst = np.hstack([a.pose.tolist()[:-1] - f_c for a in self.agents])
        hed = np.hstack(FormEnv.cossin(a.pose.theta) for a in self.agents)
        return obs, cst, hed

    def step(self, actions):
        assert self.goal is not None
        for agent_id, action in actions.items():
            [self.agents[agent_id].step(action) for _ in range(self.num_iter)]
        new_obs = self.compute_obs()
        return (*new_obs, *self.compute_reward())

    def compute_reward(self):
        reward, done = -self.step_penalty, False
        sides = sorted([Pose.dist(i.pose, j.pose) for i in self.agents 
                        for j in self.agents if i.id < j.id])
        if (sides - np.array(self.goal_sides) < 0.15).all():
            reward, done = self.max_reward, True
        print(sides - np.array(self.goal_sides))
        return reward, done, {"success": done}

