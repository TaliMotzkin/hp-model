

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

args = parse_args()
seq = args.seq
seed = args.seed
algo = args.network_choice
network_choice = args.network_choice
num_episodes = args.num_episodes
agent_choice = args.agent_choice
buffer = args.buffer
update_timestep = args.update_timestep
K_epochs = args.K_epochs               # update policy for K epochs in one PPO update

base_dir = f"./{datetime.datetime.now().strftime('%m%d-%H%M')}-"
config_str = f"{seq[:6]}-{algo}-{agent_choice}-{seed}-{num_episodes}-not_fixing_action"
save_path = base_dir + config_str + "/"
writer = SummaryWriter(f"logs/{save_path}")

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

print("seq = ", seq)
print("seed = ", seed)
print("algo = ", algo)
print("save_path = ", save_path)
print("num_episodes = ", num_episodes)

max_steps_per_episode = len(seq)



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


# NOTE: partial_reward Sep15 changed to delta of curr-prev rewards
env = gym.make("HPEnv_v0", seq=seq)

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
col_length = len(seq)


state_dim = row_width
eps_clip = 0.2


ppo_agent = PPO(state_dim, n_actions, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device, algo, writer)

# time the experiment
start_time = time()


for n_episode in tqdm(range(num_episodes)):


    s = env.reset()
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

        print("done_mask", done_mask)
        # saving reward and is_terminals
        ppo_agent.buffer.rewards.append(r)
        ppo_agent.buffer.is_terminals.append(done_mask)


        s_prime = one_hot_state(s_prime[0], s_prime[1])
        s = s_prime

        score += r

        # check if the last action ended the episode
        if done:
            break

    print("ppo_agent.buffer.is_terminals", ppo_agent.buffer.is_terminals, len(ppo_agent.buffer.is_terminals))
    # update PPO agent
    if n_episode % update_timestep == 0 and n_episode > 0:
        # print("updating")
        ppo_agent.update(n_episode, num_episodes)



    # Add current episode reward to total rewards list
    rewards_all_episodes[n_episode] = score
    # Add episodic reward onto Tensorboard
    writer.add_scalar("Reward (Episode)", score, n_episode)
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



