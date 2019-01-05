import numpy as np
from collections import deque

class Memory:
    def __init__(self, maxlen, seed=None):
        self.episodes = deque(maxlen=maxlen)
        if seed is not None:
            np.random.seed(seed)
        self.num_objects = None

    def add(self, episode):
        if self.num_objects is None:
            self.num_objects = episode.num_objects
        self.episodes.append(episode)

    def sample(self, num, tracelen):
        # returns a list of objects each of dim [num, tracelen, n_dim]
        samples = np.random.choice(self.size, num, True)
        batches = zip(*[self.episodes[i].sample(tracelen) for i in samples])
        return list(map(np.array, batches))

    @property
    def size(self):
        return len(self.episodes)

class Episode:
    def __init__(self, num_objects):
        self.num_objects = num_objects
        self.data = [[] for i in range(num_objects)]

    def add(self, experience):
        [self.data[i].append(j) for i, j in enumerate(experience)]

    def sample(self, tracelength):
        cert = []
        i = np.random.randint(self.size)
        end = i + tracelength
        overflow = end - self.size
        for x in self.data:
            data = []
            for j in range(i, end):
                try:
                    data.append(x[j])
                except:
                    data.append(x[self.size-1])
            cert.append(data)
        return cert

    @property
    def size(self):
        return len(self.data[0])
    