from collections import OrderedDict
from enum import Enum

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Action(Enum):
    LEFT = 0
    FORWARD = 1
    RIGHT = 2


class HPEnv(gym.Env):
    def __init__(self, seq):
        self.seq = seq.upper()
        self.seq_len = len(self.seq)
        self.actions = []

        self.reset()

        if len(self.seq) <= 2:
            raise ValueError("len(seq) must be > 2")

        # Note that the action space is 
        # 0 = Left, 1 = Forward, 2 = Right
        self.action_space = spaces.Discrete(3)
        # The observation space is the series of actions taken
        # 0 = Left, 1 = Forward, 2 = Right, 3 = No action,
        self.observation_space = spaces.Box(low=0, high=3, shape=(self.seq_len - 2,), dtype=np.uint8)

        # Whether the first left turn left has been taken
        # False after environment reset
        self.first_turn_left = False

    def observe(self):
        observation = np.ones(shape=(self.seq_len - 2,), dtype=np.uint8) * 4

        for i, action in enumerate(self.actions):
            observation[i] = action.value

        return observation

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        self.actions = []
        self.last_action = None
        self.prev_reward = 0
        self.state = OrderedDict(
            {
                (0, 0): self.seq[0],
                (0, 1): self.seq[1],
            }
        )
        self.done = len(self.seq) == 2
        observation = self.observe()
        self.first_turn_left = False

        return observation, {}
