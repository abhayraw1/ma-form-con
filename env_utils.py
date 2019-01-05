import numpy as np
from numpy import cos, sin
from numpy.linalg import norm
from numpy.random import random

from util import error, log

def wrap_angle(angle):
    angle = angle % (2*np.pi)
    return angle - 2*np.pi if angle > np.pi else angle

class Agent:
    def __init__(self, _id, pose=None, radius=0.15):
        self.id = _id
        self.r = radius
        self.reset(pose)

    def reset(self, pose=None):
        if pose is None:
            pose = Pose()
        self.pose = pose

    def step(self, action, dt=0.01):
      if np.array(action).any():
        self.pose.step(action, dt)

    def __str__(self):
      info = "AGENT {}:\n-Pose: {}\n-Radius: {}\n".\
              format(self.id, self.pose, self.r)
      return info

    def observes(self, oa):
        return self.pose.observes(oa.pose)

class Pose:
    def __init__(self, x=0, y=0, t=0):
        self.update(x, y, t)

    def update(self, x, y, t):
        self.x = x
        self.y = y
        self.t = t
        self.c = cos(t)
        self.s = sin(t)

    def step(self, action, dt):
        v, w = action
        x = self.x + v*self.c*dt
        y = self.y + v*self.s*dt
        t = self.t + w*dt
        self.update(x, y, t)

    def observes(self, b):
        x = (b.x - self.x)*self.c + (b.y - self.y)*self.s
        y = (b.y - self.y)*self.c - (b.x - self.x)*self.s
        h = b.c*self.c + b.s*self.s
        k = b.s*self.c - b.c*self.s
        return np.array([x, y, h, k])

    def asPoint(self):
        return [self.x, self.y]
