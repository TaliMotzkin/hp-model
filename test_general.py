import datetime
import os  # for creating directories
import pickle
import random
import sys
from collections import deque
from time import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from envs import Action
from models import *
from config import *
from training_loop_General import one_hot_state

def test_trained_model(seed,writer,  trained_model, test_seq, num_episodes=10):

    env = gym.make("HPEnvGeneral", seq=test_seq)
    env.action_space.seed(seed)

    total_rewards = []

    for episode in range(num_episodes):
        s = env.reset(test_seq[0], test_seq[1])
        s = one_hot_state(s[0][0], s[0][1])
        done = False
        score = 0.0

        while not done:
            with torch.no_grad():
                a = trained_model(s.float().unsqueeze(0)).argmax().item()
            (s_prime, r, terminated, truncated, info) = env.step(a)


            done = terminated or truncated
            s_prime = one_hot_state(s_prime[0], s_prime[1])

            s = s_prime
            score += r

        total_rewards.append(score)
        print(f"Test Episode {episode}: Score = {score:.2f}")
        writer.add_scalar("Reward (Episode)", score, episode)

    avg_test_reward = np.mean(total_rewards)
    print(f"Average Test Reward: {avg_test_reward:.2f} / 1.0")
    return avg_test_reward


hp_depth = 2  # {H,P} binary alphabet
action_depth = 4  # 0,1,2,3 in observation_box
row_width = action_depth + hp_depth
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_actions = 4
args = parse_args()
seed = args.seed
network_choice = args.network_choice
test_sequence = args.test_sequence
agent_choice = args.agent_choice
num_episodes = args.num_episodes

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

if network_choice == "AutoMaskLSTM":
    # config for RNN
    input_size = row_width
    # number of nodes in the hidden layers
    hidden_size = 256
    num_layers = 2

    print("AutoMaskLSTM with:")
    print(
        f"inputs_size={input_size} hidden_size={hidden_size} num_layers={num_layers} num_classes={n_actions}"
    )
    # Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)
    q = RNN_LSTM_onlyLastHidden(input_size, hidden_size, num_layers, n_actions, device).to(
        device
    )
    q_target = RNN_LSTM_onlyLastHidden(
        input_size, hidden_size, num_layers, n_actions, device
    ).to(device)
elif network_choice == "MambaModel":
    # config for RNN
    input_size = row_width
    # number of nodes in the hidden layers
    hidden_size = 64
    num_layers = 2

    print("MambaModel with:")
    print(
        f"inputs_size={input_size} hidden_size={hidden_size} num_layers={num_layers} num_classes={n_actions}"
    )
    # Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)
    q = MambaModel(input_size, hidden_size, num_layers, n_actions, device).to(
        device
    )
    q_target = MambaModel(
        input_size, hidden_size, num_layers, n_actions, device
    ).to(device)


base_dir = f"./{datetime.datetime.now().strftime('%m%d-%H%M')}-"
config_str = f"{test_sequence}-{network_choice}-{agent_choice}-{seed}-{num_episodes}"
save_path_new = base_dir + config_str + "/"

saved_path_model = args.saved_path_model



q.load_state_dict(torch.load(saved_path_model))
q.eval()  # Set to evaluation mode


writer = SummaryWriter(f"logs/{save_path_new}")

# Run test
test_reward = test_trained_model(seed,writer, q,test_sequence, num_episodes)
