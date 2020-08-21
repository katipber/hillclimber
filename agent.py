import heapq
import os
import time
from collections import deque
from copy import deepcopy as dc
from math import sqrt

import numpy as np
import pandas as pd
import torch
from scipy import stats


class Node:
    _id = 0

    def __init__(self, agent, model, ci=.9):
        self.id = Node.id()

        self.agent = agent
        self.model = model

        self.ci = ci

        self.scores = deque(maxlen=agent.max_run)

        self._mean = None
        self.ci_low = None

        self._key = None
        self.set_key()

        self.n_expand = 0
        self.max_expand = self.key_size * self.agent.n_steps * 2 * 2

    @staticmethod
    def id():
        Node._id += 1
        return Node._id - 1

    def __lt__(self, other):
        if self.mean != other.mean:
            return self.ci_low < other.ci_low
        return self.mean < other.mean

    @property
    def runs(self):
        return len(self.scores)

    @property
    def mean(self):
        return np.mean(self.scores) if self._mean is None else self._mean

    @property
    def weights(self):
        return self.model.state_dict()

    @weights.setter
    def weights(self, new_weights):
        self.model.load_state_dict(new_weights)
        self.model.eval()
        self.set_key()

    @property
    def key(self):
        return self._key

    @key.setter
    def key(self, new_key):
        weights = dict()

        for k, v in self.weights.items():
            weight = new_key[:v.numel()]
            new_key = new_key[v.numel():]

            weights[k] = torch.tensor(weight).reshape(v.shape)

        self.weights = weights

    @property
    def key_size(self):
        return len(self._key)

    @property
    def random_key(self):
        key = [np.random.rand() for _ in range(self.key_size)]
        key *= self.dirs(True)
        return key

    @property
    def can_expand(self):
        return self.n_expand < self.max_expand

    @property
    def next_key(self):
        key = self.key * (1 + self.delta())
        self.n_expand += 1
        return key

    @property
    def random(self):
        return self.n_expand % 2 == 1

    def delta(self, mask=None, step=None, dirs=None):
        mask = self.mask(True) if mask is None else mask
        step = self.step(True) if step is None else step
        dirs = self.dirs(True) if dirs is None else dirs

        return np.multiply(mask, step) * dirs

    def mask(self, random=False):
        if random or self.random:
            return np.random.choice([0, 1], self.key_size)

        index = self.n_expand // 2 // 2 % self.key_size

        align = ':0>'
        mask_str = f'{{{align}{self.key_size}b}}'
        mask = mask_str.format(2 ** index)
        mask = [int(i) for i in mask[::-1]]

        return mask

    def step(self, random=False):
        if random or self.random:
            return np.random.choice(self.agent.steps, self.key_size) * np.random.rand(self.key_size)

        index = self.n_expand // 2 // 2 // self.agent.n_steps

        return self.agent.steps[index]

    def dirs(self, random=False):
        if random or self.random:
            return np.random.choice([-1, 1], self.key_size)

        index = self.n_expand // 2 % 2

        return -1 ** index

    def set_key(self):
        key = []

        for v in self.weights.values():
            key.extend(v.flatten().tolist())

        self._key = tuple(key)

    def value(self, state):
        with torch.no_grad():
            return self.model(state)

    def update_score(self, score):
        self.scores.append(score)

    def update_stats(self):
        n = self.runs
        m = np.mean(self.scores)

        t = stats.t.ppf(self.ci, n - 1)

        sd = np.std(self.scores, ddof=1)

        s = sd / sqrt(n)

        err = t * s

        self._mean = m
        self.ci_low = m - err

    def add_noise(self):
        self.key = self.key + self.key * self.mask(True) * min(self.agent.steps) * self.dirs(True)
        self.model.eval()

    def clone(self, key):
        model = dc(self.model)
        node = Node(self.agent, model, ci=self.ci)

        node.key = self.random_key if key is None else key
        node.model.eval()

        return node


class HillClimber:
    header = ['test', 'time', 'epoch',
              'node_id', 'node_score', 'node_mean',
              'seed_id', 'seed_mean', 'seed_ci_low', 'seed_expansion',
              'best_id', 'best_mean', 'best_ci_low']

    def __init__(self, model, steps=None, ci=.9, min_run=20, max_run=100, batch_size=5, test=False, file_name=None,
                 file_path=None):
        self.steps = [2 ** s for s in range(-5, 6)] if steps is None else steps

        self.test = test
        self.min_run = min_run
        self.max_run = max_run
        self.batch_size = batch_size

        self._node = Node(self, model, ci)
        self.seed = None
        self.best = self._node

        self.batch = [self.node]
        self.epoch = 0

        self._path = './' if file_path is None else file_path
        self._file = time.strftime('%d-%m-%y_%X', time.gmtime()) if file_name is None else file_name

        self.log_header = HillClimber.header + [i for i in range(self.node.key_size)]
        self.reset_log()

    @property
    def file_path(self):
        return self._path

    @file_path.setter
    def file_path(self, path):
        self._path = path
        self.reset_log()

    @property
    def file_name(self):
        return self._file

    @file_name.setter
    def file_name(self, name):
        self._file = name
        self.reset_log()

    @property
    def node(self):
        if self.test:
            return self.best

        return self._node

    @node.setter
    def node(self, new_node):
        self._node = new_node

    @property
    def n_steps(self):
        return len(self.steps)

    @staticmethod
    def is_better(a, b):
        a.update_stats()

        if a.mean == b.mean:
            return a.ci_low > b.ci_low

        return a.mean > b.mean

    def eval(self, state):
        value = self.value(state)
        return torch.argmax(value).item()

    def value(self, state):
        state = torch.tensor(state, dtype=torch.float)
        return self.node.value(state)

    def update_score(self, score):
        self.node.update_score(score)

        log = self.save_log()

        if self.test:
            self.node.update_stats()
        else:
            self.update_node()

        return log

    def update_node(self):
        if self.seed is None:
            if self.node.runs >= self.min_run:
                self.update_seed()
                self.update_batch()
                self.update_best()
                self.expand()

        elif self.node.runs >= self.min_run:
            self.update_seed()
            self.update_batch()
            self.update_best()
            self.expand()

        elif self.node.mean <= self.seed.ci_low:
            self.expand()

    def update_best(self):
        if self.is_better(self.node, self.best):
            self.best = self.node

    def update_batch(self):
        if self.is_better(self.node, self.batch[0]):
            heapq.heappush(self.batch, self.node)

        if len(self.batch) > self.batch_size:
            heapq.heappop(self.batch)

    def update_seed(self):
        if self.seed is None or self.is_better(self.node, self.seed):
            self.seed = self.node

    def expand(self):
        while self.seed.can_expand:
            key = self.seed.next_key
            if any(key):
                self.node = self.node.clone(key)
                return

        self.sample()

    def sample(self):
        self.epoch += 1
        self.seed = None

        if len(self.batch) == 1:
            self.node = self.node.clone(self.node.random_key)
            return

        batch_mean = np.mean([n.mean for n in self.batch])
        self.batch = [n for n in self.batch if n.mean > batch_mean]
        batch_mean = np.mean([n.mean for n in self.batch])

        keys = []
        if len(self.batch) > 0:
            for n in self.batch:
                keys.append(np.multiply(n.key, n.mean) / batch_mean)
        else:
            keys.append(self.best.key)
            self.batch.append(self.best)

        key = np.mean(keys, 0)

        self.node = self.node.clone(key)
        self.node.add_noise()

        heapq.heapify(self.batch)

    def save_model(self, file_name=None, file_path=None):
        if file_name is None:
            file_name = self._file + '.model'

        if file_path is None:
            file_path = self._path

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        torch.save(self.best.weights, os.path.join(file_path, file_name))

    def load_model(self, file_name=None, file_path=None):
        if file_name is None:
            file_name = self._file + '.model'

        if file_path is None:
            file_path = self._path

        weights = torch.load(os.path.join(file_path, file_name))

        self.best.weights = weights

    def reset_log(self):
        file_name = self._file + '.log'

        df = pd.DataFrame(columns=self.log_header)
        df.to_csv(os.path.join(self._path, file_name), mode='w', index=False)

    def save_log(self):
        epi_log = [self.test, time.time(), self.epoch]
        node_log = [self.node.id, self.node.scores[-1], self.node.mean]
        seed_log = [None, None, None, None] if self.seed is None else \
            [self.seed.id, self.seed.mean, self.seed.ci_low, self.seed.n_expand / self.seed.max_expand]
        best_log = [None, None, None] if self.best is None else \
            [self.best.id, self.best.mean, self.best.ci_low]

        log = [epi_log + node_log + seed_log + best_log + list(self.node.key)]

        file_name = self._file + '.log'

        df = pd.DataFrame(log)
        df.to_csv(os.path.join(self._path, file_name), mode='a', index=False, header=False)

        return df
