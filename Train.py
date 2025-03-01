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
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
from envs import Action

from Agents.Costum_DQN import DQNAgent
from torch.utils.tensorboard import SummaryWriter



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
    writer = SummaryWriter(log_dir='./tensorboard_logs')
    max_rewards_per_episode = []
    mean_rewards_per_episode = []

    start_time = time.time()
    episode_step = 0
    for episode in range(episodes):
        state, _ = env.reset()
        # if episode_step < 5:
        #     print("state", state)
        state_actions = torch.tensor(state[0], dtype=torch.float32).to(device)
        state_seq = torch.tensor(state[1], dtype=torch.float32).to(device)

        agent.q_network.reset_hidden(batch_size=1)
        done = False

        episode_rewards = []  # Track all rewards in this episode

        while not done:
            action = agent.select_action_1(state_actions, state_seq)
            next_state, reward, done, _, _ = env.step(action)
            # if episode_step < 5:
            #     print("state", next_state)

            next_state_actions = torch.tensor(next_state[0], dtype=torch.float32).to(device)
            next_state_seq = torch.tensor(next_state[1], dtype=torch.float32).to(device)


            agent.buffer.push(state_actions, state_seq, action, reward, next_state_actions, next_state_seq, done)
            agent.train_step()

            state_actions = next_state_actions
            state_seq = next_state_seq
            episode_rewards.append(reward)  # Store reward per timestep

        # Compute statistics for the episode
        max_reward = max(episode_rewards)  # Max reward in this episode
        mean_reward = np.mean(episode_rewards)  # Mean reward in this episode

        max_rewards_per_episode.append(max_reward)
        mean_rewards_per_episode.append(mean_reward)

        writer.add_scalar('Max Reward/Episode', max_reward, episode)
        writer.add_scalar('Mean Reward/Episode', mean_reward, episode)

        # Decay epsilon
        agent.epsilon = agent.epsilon_min + (agent.epsilon_max - agent.epsilon_min)* np.exp(-episode_step*5/episodes)

        # Update target network every X episodes
        if episode % target_update == 0:
            agent.update_target_network()
        if episode % 1000 == 0:
            time_episode = time.time() - start_time
            print(f"Episode {episode + 1}, Max Reward: {max_reward}, Mean Reward: {mean_reward:.3f}, Epsilon: {agent.epsilon:.3f} at time {time_episode}")

        episode_step += 1
    end_time = time.time()
    print(f"Total time: {end_time - start_time}")
    plot_episode_rewards(max_rewards_per_episode, mean_rewards_per_episode)
    writer.close()

# Automatically detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(device, seed=42):
    """Ensure reproducibility by setting the random seed globally"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure deterministic behavior in CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable optimizations that may introduce randomness

    seq = "PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP"
    env = gym.make('HPEnv_v0', seq=seq)
    env.reset(seed=seed)

    print(f" Random Seed Set: {seed}")
    return env

seed = 0

env = set_seed(device, seed)
agent = DQNAgent(env)

train_dqn(env, agent, episodes=250000)