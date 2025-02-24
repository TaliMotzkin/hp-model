from enum import Enum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Action(Enum):
    LEFT = 0
    FORWARD = 1
    RIGHT = 2
    NONE = 3


class HPEnv(gym.Env):
    def __init__(self, seq):
        self.seq = seq.upper()
        self.seq_len = len(self.seq)
        self.actions = []

        observation_sequence = []
        for character in self.seq:
            if character == 'H':
                observation_sequence.append(0)
            elif character == 'P':
                observation_sequence.append(1)
        self.observation_sequence = np.array(observation_sequence, dtype=np.uint8)

        self.reset()

        if len(self.seq) <= 2:
            raise ValueError("len(seq) must be > 2")

        # Note that the action space is 
        # 0 = Left, 1 = Forward, 2 = Right
        self.action_space = spaces.Discrete(3)
        # The observation space is the series of actions taken
        # 0 = Left, 1 = Forward, 2 = Right, 3 = No action,
        self.observation_space = spaces.Tuple((spaces.Box(low=0, high=3, shape=(self.seq_len,), dtype=np.uint8),
                                               spaces.Box(low=0, high=1, shape=(self.seq_len,), dtype=np.uint8)))

        # Whether the first left turn left has been taken
        # False after environment reset
        self.first_turn_left = False

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("%r (%s) invalid" % (action, type(action)))

        # Force the first turning action to be Left
        if (action != 1) and (self.first_turn_left is False):
            if action == 2:
                action = 0
            self.first_turn_left = True

        prev_pos = self.state[-1]
        prev_prev_pos = self.state[-2]
        direction_order = [(0, 1), (1, 0), (0, -1), (-1, 0)] # up, right, down, left
        cur_direction = (prev_pos[0] - prev_prev_pos[0], prev_pos[1] - prev_prev_pos[1])
        cur_direction_idx = direction_order.index(cur_direction)
        if action == Action.LEFT.value:
            cur_direction_idx = (cur_direction_idx - 1) % 4  # Turn left 
        elif action == Action.RIGHT.value:
            cur_direction_idx = (cur_direction_idx + 1) % 4  # Turn right
        direction = direction_order[cur_direction_idx]
        new_pos = (prev_pos[0] + direction[0], prev_pos[1] + direction[1])

        observation = self.observe()

        # Detects for collision
        if new_pos in self.state:
            return (observation, 0, True, False, {})

        self.actions.append(action)
        self.state.append(new_pos)

        self.terminated = len(self.state) == self.seq_len
        self.truncated = False # Always False
        reward = self._compute_reward()
        info = {
            'chain_length' : len(self.state),
            'seq_length'   : self.seq_len,
            'actions'      : self.actions,
            'state'  : self.state,
            'first_turn_left': self.first_turn_left,
        }

        return (observation, reward, self.terminated, self.truncated, info)

    def _compute_reward(self) -> float:
        # The reward is the number of H-H non-sequential contacts
        num_contacts = 0.
        h_positions = {pos for idx, pos in enumerate(self.state) if self.seq[idx] == 'H'}

        # Check adjacent H-H interactions
        for x, y in h_positions:
            neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            for nx, ny in neighbors:
                # Second condition checks if non-sequential
                if (nx, ny) in h_positions and self.state.index((nx, ny)) not in {self.state.index((x, y)) - 1, self.state.index((x, y)) + 1}:
                    num_contacts += 1

        return num_contacts / 2  # Each pair is counted twice


    def observe(self):
        observation_actions = np.ones(shape=(self.seq_len,), dtype=np.uint8) * 3

        for i, action in enumerate(self.actions):
            observation_actions[i + 2] = action

        return (observation_actions, self.observation_sequence)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        self.actions = []
        self.prev_reward = 0
        self.state = [ (0, 0), (0, 1) ]
        self.terminated = len(self.seq) == 2
        observation = self.observe()
        self.first_turn_left = False

        return observation, {}
