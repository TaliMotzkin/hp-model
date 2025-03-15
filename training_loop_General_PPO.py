
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
from PPO import PPO
import pandas as pd


args = parse_args()
directory_data = args.seq_list
df = pd.read_pickle(directory_data)
seq_list = list(zip(df["HP Sequence"], df["Best Known Energy"]))
seq_lengths = list(df["Length"])[args.start_learning: args.stop_learning]

seq_list = seq_list[args.start_learning: args.stop_learning]
max_seq_length = len(seq_list[-1][0])
seed = args.seed
algo = args.network_choice
network_choice = args.network_choice
num_episodes = args.num_episodes
agent_choice = args.agent_choice
buffer = args.buffer
update_timestep = args.update_timestep
K_epochs = args.K_epochs               # update policy for K epochs in one PPO update
use_curriculum = args.use_curriculum  # Set to False for purely random learning
revisit_probability = args.revisit_probability  # Probability of revisiting an earlier sequence in curriculum mode





# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
gamma = 0.98  # discount rate
lr_actor = 0.0005
lr_critic = 0.0001


print("##### Summary of Hyperparameters #####")
print("learning_rate_actor: ", lr_actor)
print("learning_rate_critic: ", lr_critic)
print("GAMMA: ", gamma)
print("update_timestep: ", update_timestep)
print("##### End of Summary of Hyperparameters #####")


# render settings
show_every = num_episodes // 2  # for plot_print_rewards_stats
# metric for evaluation
rewards_all_episodes = np.zeros(
    (num_episodes,),
    # dtype=np.int32
)
reward_max = 0
best_folds = []
# keep track of trapped SAW
num_trapped = 0

warmRestart = True
decay_mode = "exponential"  # exponential, cosine, linear
num_restarts = 1  # for cosine decay warmRestart=True
exploration_decay_rate = 5  # for exponential decay
start_decay = 0  # for exponential and linear
print(f"decay_mode={decay_mode} warmRestart={warmRestart}")
print(
    f"num_restarts={num_restarts} exploration_decay_rate={exploration_decay_rate} start_decay={start_decay}"
)


# Nov30 2021 add one more column of step_E
hp_depth = 2  # {H,P} binary alphabet
action_depth = 4  # 0,1,2,3 in observation_box

def one_hot_state(observation_actions, observation_sequence):
    one_hot_actions = F.one_hot(torch.from_numpy(observation_actions), num_classes=4)
    one_hot_sequence = F.one_hot(torch.from_numpy(observation_sequence), num_classes=2)
    one_hot_input = torch.cat([one_hot_actions, one_hot_sequence], dim=-1).float()
    return one_hot_input

current_seq_idx = 0  # start from the easiest seq
progress_threshold = args.progress_threshold  # move to next sequence when reward reaches 80% of optimal
# moving_avg_window = 100  # wondpw size for checking progression
# recent_rewards = deque(maxlen=moving_avg_window)  # Track last rewards

# NOTE: partial_reward Sep15 changed to delta of curr-prev rewards
env = gym.make("HPEnvGeneral", seq=seq_list[0][0], maximal=seq_list[0][1])

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
env.action_space.seed(seed)

# initial state/observation
# NOTE: env.state != state here
# env.state is actually the chain of OrderedDict([((0, 0), 'H')])
# the state here actually refers to the observation!
initial_state = env.reset()

print("initial state/obs:")
print(initial_state)

# Get number of actions from gym action space
n_actions = env.action_space.n
print("n_actions = ", n_actions)

row_width = action_depth + hp_depth


state_dim = row_width
eps_clip = 0.2


base_dir = f"./{datetime.datetime.now().strftime('%m%d-%H%M')}-"
pre_trained_dir = args.pre_trained_model

if args.pre_trained:
    base_dir_pre = os.path.basename(os.path.dirname(pre_trained_dir))
    timestamp = "-".join(base_dir_pre.split("-")[:2])
    config_str = f"{algo}-{agent_choice}-{seed}-{num_episodes}-not_fixing_action-{args.start_learning}-{args.stop_learning}-{use_curriculum}-Pre_trained_on-{timestamp}"
else:
    config_str = f"{algo}-{agent_choice}-{seed}-{num_episodes}-not_fixing_action-{args.start_learning}-{args.stop_learning}-{use_curriculum}"

save_path = base_dir + config_str + "/"
writer = SummaryWriter(f"logs/{save_path}")

ppo_agent = PPO(state_dim, n_actions, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device, algo, writer, max_seq_length)
if args.pre_trained:
    ppo_agent.policy.load_state_dict(torch.load(pre_trained_dir))
    ppo_agent.policy_old.load_state_dict(torch.load(pre_trained_dir))


# whether to show or save the matplotlib plots
display_mode = "save"  # save for CMD, show for ipynb
if display_mode == "save":
    save_fig = True
else:
    save_fig = False

# create the folder according to the save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)


# apply flush=True to every print function call in the module with a partial function
from functools import partial

print = partial(print, flush=True)

print("seq_list = ", seq_list)
print("seed = ", seed)
print("algo = ", algo)
print("save_path = ", save_path)
print("num_episodes = ", num_episodes)

# time the experiment
start_time = time()

rewards_per_sequence_length = {}
for n_episode in tqdm(range(num_episodes)):

    if use_curriculum:
        if np.random.rand() < revisit_probability and current_seq_idx > 0:
            sampled_idx = np.random.randint(0, current_seq_idx + 1)
        else:
            sampled_idx = current_seq_idx
    else:
        # randomly pick any sequence
        sampled_idx = np.random.randint(0, len(seq_list))

    # reset the environment
    # Initialize the environment and state
    current_seq, opt_reward = seq_list[sampled_idx]
    s = env.reset(options={"new_seq": current_seq, "maximal": opt_reward})
    max_steps_per_episode = len(current_seq)

    s = one_hot_state(s[0][0], s[0][1])

    done = False
    score = 0.0

    # whether to avoid F in the next step?
    avoid_F = False

    for step in range(max_steps_per_episode):
        # unsqueeze(0) adds a dimension at 0th for batch=1
        # i.e. adds a batch dimension

        # select action with policy
        a = ppo_agent.select_action(s.float().unsqueeze(0))
        (s_prime, r, terminated, truncated, info) = env.step(a)

        # a = info["actions"][-1]
        done = terminated or truncated
        done_mask = 1.0 if done else 0.0

        # saving reward and is_terminals
        ppo_agent.buffer.rewards.append(r)
        ppo_agent.buffer.is_terminals.append(done_mask)


        s_prime = one_hot_state(s_prime[0], s_prime[1])
        s = s_prime

        score += r

        # check if the last action ended the episode
        if done:
            break

    # recent_rewards.append(score)
    # update PPO agent
    if n_episode % update_timestep == 0 and n_episode > 0:
        # print("updating")
        ppo_agent.update(n_episode, num_episodes)

    # Check if its time to move to the next sequence
    if use_curriculum and n_episode > 200 and current_seq_idx < len(seq_list) - 1:

        key = (current_seq[:6], max_steps_per_episode)
        # print("key", key)
        # if key in rewards_per_sequence_length:
        #     print("len,", len(rewards_per_sequence_length[key]))
        if key in rewards_per_sequence_length and len(rewards_per_sequence_length[key]) == 200 and current_seq_idx == sampled_idx:

            avg_reward = np.mean(rewards_per_sequence_length[key]) / seq_list[sampled_idx][1]  # Normalize by max reward

            writer.add_scalar("Curriculum Learning: Normalized Average Reward (Episode)", avg_reward, n_episode)

            if avg_reward >= progress_threshold:
                print(f"Moving to next sequence {current_seq_idx + 1} (Avg Normalized Reward: {avg_reward:.2f})")
                current_seq_idx += 1
                rewards_per_sequence_length.clear()

                # Add current episode reward to total rewards list
    rewards_all_episodes[n_episode] = score
    # Add episodic reward onto Tensorboard
    writer.add_scalar("Reward (Episode)", score, n_episode)
    writer.add_scalar("Sequence Length (Episode)", max_steps_per_episode, n_episode)

    # Adding real reward per each sequence
    for seq in seq_list:
        if seq[0] == current_seq:
            reward_value = score * seq[1]
            writer.add_scalar \
                (f"Sequence: {current_seq[:6]}, Length: {max_steps_per_episode}, Maximal Reward: {seq[1]} - Reward (Episode)", reward_value, n_episode)

            key = (current_seq[:6], max_steps_per_episode)
            if key not in rewards_per_sequence_length:
                rewards_per_sequence_length[key] = []

            rewards_per_sequence_length[key].append(reward_value)

            if len(rewards_per_sequence_length[key]) > 200:
                rewards_per_sequence_length[key].pop(0)
            writer.add_scalar \
                (f"Sequence: {current_seq[:6]}, Length: {max_steps_per_episode}, Maximal Reward: {seq[1]} Reward (Episode) - Reward (Windowed Max)",
                              np.max(rewards_per_sequence_length[key]), n_episode)

    # Add window of max episodic reward over the last 200 episodes onto Tensorboard
    if n_episode > 200:
        writer.add_scalar("Reward (Episode Windowed)", np.max(rewards_all_episodes[n_episode - 200 : n_episode]), n_episode)
    # update max reward found so far
    if score > reward_max:
        print("found new highest reward = ", score)
        reward_max = score
        best_folds.append(info)
        print(info)

    if (n_episode == 0) or ((n_episode + 1) % show_every == 0):
        print(
            "Episode {}, score: {:.1f}, reward_max: {}".format(
                n_episode,
                score,

                reward_max,
            )
        )
        print(
            f"\ts_prime: {s_prime[:3], s_prime.shape}, reward: {r}, done: {done}, info: {info}"
        )
    # move on to the next episode

print("Complete")
# for time records
end_time = time()
elapsed = end_time - start_time
print("Training time: ", elapsed)

# Save the rewards_all_episodes with numpy save
with open(f"{save_path}{config_str}-rewards_all_episodes.npy", "wb") as f:
    np.save(f, rewards_all_episodes)

# Save the best foldings
with open(f"{save_path}{config_str}-best_folds", "wb") as f:
    pickle.dump(best_folds, f, protocol=pickle.HIGHEST_PROTOCOL)
    # Print the best foldings
    print("Best foldings:")
    print(best_folds)


ppo_agent.save(f"{save_path}{config_str}-state_dict.pth")
env.close()



