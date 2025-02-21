import gymnasium as gym
import os
import random
import torch
from models.RNN import RNN_class
from Agents.DQN_RNN import CUSTOM
# import the skrl components to build the RL system
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.models.torch import deterministic
from skrl.utils.model_instantiators.torch import Shape, deterministic_model

import matplotlib.pyplot as plt
import numpy as np
from envs import Action

from Agents.Costum_DQN import DQNAgent



import matplotlib.pyplot as plt
import numpy as np
import time


def plot_episode_rewards(max_rewards, mean_rewards):
    """
    Plots:
    - Maximum reward per episode
    - Average reward per episode

    Args:
    - max_rewards (list): List of max rewards per episode.
    - mean_rewards (list): List of mean rewards per episode.
    """

    episodes = np.arange(len(max_rewards))  # X-axis: Episodes

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, max_rewards, label="Max Reward Per Episode", color="red", linestyle='solid')
    plt.plot(episodes, mean_rewards, label="Mean Reward Per Episode", color="blue", linestyle='dashed')

    # Labels and title
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Max and Mean Rewards per Episode")
    plt.legend()
    plt.grid()
    plt.savefig("episode_rewards.png")

    print(f"Maximum reward achieved in any episode: {max(max_rewards)}")


def train_dqn(env, agent, episodes=1000, target_update=10):
    max_rewards_per_episode = []
    mean_rewards_per_episode = []

    start_time = time.time()
    episode_step = 0
    for episode in range(episodes):
        state, _ = env.reset()
        # if episode_step < 5:
        #     print("state", state)
        state = torch.tensor(state, dtype=torch.float32).to(device)
        agent.q_network.reset_hidden(batch_size=1)
        done = False

        episode_rewards = []  # Track all rewards in this episode

        while not done:
            action = agent.select_action_1(state)
            next_state, reward, done, _, _ = env.step(action)
            # if episode_step < 5:
            #     print("state", next_state)

            next_state = torch.tensor(next_state, dtype=torch.float32).to(device)

            agent.buffer.push(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            episode_rewards.append(reward)  # Store reward per timestep

        # Compute statistics for the episode
        max_reward = max(episode_rewards)  # Max reward in this episode
        mean_reward = np.mean(episode_rewards)  # Mean reward in this episode

        max_rewards_per_episode.append(max_reward)
        mean_rewards_per_episode.append(mean_reward)

        # Decay epsilon
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

        # Update target network every X episodes
        if episode % target_update == 0:
            agent.update_target_network()
        if episode % 100 == 0:
            time_episode = time.time() - start_time
            print(f"Episode {episode + 1}, Max Reward: {max_reward}, Mean Reward: {mean_reward:.3f}, Epsilon: {agent.epsilon:.3f} at time {time_episode}")

        episode_step += 1
    end_time = time.time()
    print(f"Total time: {end_time - start_time}")
    plot_episode_rewards(max_rewards_per_episode, mean_rewards_per_episode)

# Automatically detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed=42):
    """Ensure reproducibility by setting the random seed globally"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure deterministic behavior in CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable optimizations that may introduce randomness

    seq = "PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH"
    env = gym.make('HPEnv_v0', seq=seq)
    env.reset(seed=seed)

    print(f" Random Seed Set: {seed}")
    return env

seed = 0
env = set_seed(seed)
agent = DQNAgent(env)
# Train the DQN Agent
r = train_dqn(env, agent, episodes=1000)