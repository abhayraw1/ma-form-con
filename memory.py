import numpy as np
from collections import deque

class Memory:
    def __init__(self, maxlen, seed=None):
        self.data = deque(maxlen=maxlen)
        if seed is not None:
            np.random.seed(seed)

    def add(self, *args):
        self.data.append(*args)

    def sample(self, num):
        if num > self.size:
            raise ValueError("Memory size: {}, but requested: {}"
                             .format(self.size, num))
        samples = np.random.choice(self.size, num, False)
        batches = [self.data[i] for i in samples]
        return batches

    @property
    def size(self):
        return len(self.data)
