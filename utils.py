from copy import deepcopy as dc

import gym
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder

import recorder

enc = OneHotEncoder()


class Environment:
    @staticmethod
    def linear_model(dims):
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def __init__(self):
        self.env = None
        self.rec = None
        self.target = None
        self._record = False

    def __del__(self):
        self.env.close()

    @property
    def model(self):
        raise NotImplemented

    @property
    def record(self):
        return self._record

    @record.setter
    def record(self, value):
        if value and not self._record:
            self.rec = recorder.get_recorder(dc(self.env))

        self._record = value

    def reset(self):
        if self._record:
            obs = self.rec.reset()
            recorder.start_recording(self.rec)
        else:
            obs = self.env.reset()

        return obs

    def step(self, action):
        obs, reward, done, info = self.rec.step(action) if self.record else self.env.step(action)

        if done and self.record:
            self.rec.close()

        return obs, reward, done, info

    # noinspection PyMethodMayBeStatic
    def play_video(self):
        return recorder.show_videos()


class CartPole(Environment):
    def __init__(self):
        super().__init__()
        self.env = gym.make('CartPole-v1')
        self.target = 475

    @property
    def model(self):
        return Environment.linear_model([4, 2])


class MountainCar(Environment):
    def __init__(self):
        super().__init__()
        self.env = gym.make('MountainCar-v0')
        self.target = -110

    @property
    def model(self):
        return Environment.linear_model([2, 3])


class LunarLander(Environment):
    def __init__(self):
        super().__init__()
        self.env = gym.make('LunarLander-v2')
        self.target = 200

    @property
    def model(self):
        return Environment.linear_model([8, 4])


def env_list():
    s = '{:10} {:20} {}'.format('CODE', 'NAME', 'LINK')
    s += '\n'
    s += '{:10} {:20} {}'.format('CP', 'CartPole-v1', 'https://gym.openai.com/envs/CartPole-v1/')
    s += '\n'
    s += '{:10} {:20} {}'.format('MC', 'MountainCar-v0', 'https://gym.openai.com/envs/MountainCar-v0/')
    s += '\n'
    s += '{:10} {:20} {}'.format('LL', 'LunarLander-v2', 'https://gym.openai.com/envs/LunarLander-v2/')

    return s


def get_env(code):
    if code == 'CP':
        return CartPole()
    elif code == 'MC':
        return MountainCar()
    elif code == 'LL':
        return LunarLander()
    raise ValueError(f'{code} is not a valid environment code!\nPlease call env_list() to see available codes.')
