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
from torch.utils.tensorboard import SummaryWriter



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
def plot_loss(mean_loss):
    episodes = np.arange(len(mean_loss)) 

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, mean_loss, label="mean loss Per Episode", color="red", linestyle='solid')
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.title("mean loss Per Episode")
    plt.legend()
    plt.grid()
    plt.savefig("episode_loss.png")

    
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def plot_loss(mean_loss, filename="episode_loss.png"):
    episodes = np.arange(len(mean_loss))
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, mean_loss, label="Mean Loss Per Episode", color="red", linestyle='solid')
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.title("Mean Loss Per Episode")
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close()

def plot_episode_rewards(max_rewards, mean_rewards, filename="episode_rewards.png"):
    episodes = np.arange(len(max_rewards))
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, max_rewards, label="Max Reward Per Episode", color="red", linestyle='solid')
    plt.plot(episodes, mean_rewards, label="Mean Reward Per Episode", color="blue", linestyle='dashed')
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Rewards Per Episode")
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close()

def train_dqn(env, agent, episodes=1000, target_update=100):
    # Initialize TensorBoard writer (logs will be saved in "runs/experiment_1")
    writer = SummaryWriter(log_dir='runs/experiment_1')

    max_rewards_per_episode = []
    mean_rewards_per_episode = []
    total_losses_per_episode = []  # List for storing mean loss per episode

    start_time = time.time()
    episode = 0
    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(agent.device)
        agent.q_network.reset_hidden(batch_size=1)
        done = False

        episode_rewards = []  # Track rewards per episode
        episode_losses = []   # Track losses per episode

        while not done:
            action = agent.select_action_1(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).to(agent.device)

            agent.buffer.push(state, action, reward, next_state, done)
            loss = agent.train_step()  # Ensure train_step() returns the loss (see earlier examples)
            if loss is not None:
                episode_losses.append(loss)
            state = next_state
            episode_rewards.append(reward)

        # Compute per-episode statistics
        max_reward = max(episode_rewards)
        mean_reward = np.mean(episode_rewards)
        mean_loss = np.mean(episode_losses) if episode_losses else 0.0

        total_losses_per_episode.append(mean_loss)
        max_rewards_per_episode.append(max_reward)
        mean_rewards_per_episode.append(mean_reward)

        # Log to TensorBoard
        writer.add_scalar('Reward/Max', max_reward, episode)
        writer.add_scalar('Reward/Mean', mean_reward, episode)
        writer.add_scalar('Loss/Mean', mean_loss, episode)
        writer.add_scalar('Agent/Epsilon', agent.epsilon, episode)

        for name, param in agent.q_network.named_parameters():
            writer.add_histogram(name, param, episode)

        # Decay epsilon
        agent.epsilon = agent.epsilon_min + (agent.epsilon_max - agent.epsilon_min) * np.exp(-(episode * agent.decay_rate) / episodes)

        # Update target network periodically
        if episode % target_update == 0:
            agent.update_target_network()
        if episode % 1000 == 0:
            print(f"Episode {episode + 1}, Max Reward: {max_reward}, Mean Reward: {mean_reward:.3f}, Mean Loss: {mean_loss:.4f}, Epsilon: {agent.epsilon:.3f}")
        episode += 1
    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")

    # Close the TensorBoard writer
    writer.close()

    # Generate and save local plots
    plot_loss(total_losses_per_episode, filename="episode_loss.png")
    plot_episode_rewards(max_rewards_per_episode, mean_rewards_per_episode, filename="episode_rewards.png")



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

    seq = "HHHPPHPHPHPPHPHPHPPH"
    env = gym.make('HPEnv_v0', seq=seq)
    env.reset(seed=seed)

    print(f" Random Seed Set: {seed}")
    return env

seed = 0
env = set_seed(seed)
agent = DQNAgent(env)
# Train the DQN Agent
r = train_dqn(env, agent, episodes=50000)