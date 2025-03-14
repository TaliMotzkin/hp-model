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
import pandas as pd

args = parse_args()
directory_data = args.seq_list
df = pd.read_pickle(directory_data)
seq_list = list(zip(df["HP Sequence"], df["Best Known Energy"]))

seed = args.seed
algo = args.network_choice
network_choice = args.network_choice
num_episodes = args.num_episodes
agent_choice = args.agent_choice
buffer = args.buffer
use_curriculum = args.use_curriculum  # Set to False for purely random learning
revisit_probability = args.revisit_probability  # Probability of revisiting an earlier sequence in curriculum mode

base_dir = f"./{datetime.datetime.now().strftime('%m%d-%H%M')}-"
config_str = f"{algo}-{agent_choice}-{seed}-{num_episodes}-{buffer}"
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

# Redirect 'print' output to a file in python
# orig_stdout = sys.stdout
# f = open(save_path + 'out.txt', 'w')
# sys.stdout = f

# apply flush=True to every print function call in the module with a partial function
from functools import partial

print = partial(print, flush=True)

print("seq_list = ", seq_list)
print("seed = ", seed)
print("algo = ", algo)
print("save_path = ", save_path)
print("num_episodes = ", num_episodes)


learning_rate = 0.0005

mem_start_train = 36 * 50  # for memory.size() start training
# mem_start_train = 50  # for memory.size() start training

TARGET_UPDATE = 100  # fix to 100

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
gamma = 0.98  # discount rate
batch_size = 32
train_times = 10  # number of times train was run in a loop

# capped at 50,000 for <=48mer
# buffer_limit = int(min(50000, num_episodes // 10))  # replay-buffer size
buffer_limit = 1100

print("##### Summary of Hyperparameters #####")
print("learning_rate: ", learning_rate)
print("BATCH_SIZE: ", batch_size)
print("GAMMA: ", gamma)
print("mem_start_train: ", mem_start_train)
print("TARGET_UPDATE: ", TARGET_UPDATE)
print("buffer_limit: ", buffer_limit)
print("train_times: ", train_times)
print("##### End of Summary of Hyperparameters #####")

# Exploration parameters
max_exploration_rate = 1
min_exploration_rate = 0.01

# render settings
show_every = num_episodes // 2  # for plot_print_rewards_stats
pause_t = 0.0
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


current_seq_idx = 0  #start from the easiest seq
progress_threshold = 0.5  #move to next sequence when reward reaches 80% of optimal
moving_avg_window = 5  #wondpw size for checking progression
recent_rewards = deque(maxlen=moving_avg_window)  # Track last rewards

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
    q = AutoMaskLSTM(input_size, hidden_size, num_layers, n_actions, device).to(
        device
    )
    q_target = AutoMaskLSTM(
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

q_target.load_state_dict(q.state_dict())

optimizer = optim.Adam(q.parameters(), lr=learning_rate)

beta_start = 0.4
beta_end = 1.0

def get_beta(current_step, num_episodes):
    return min(beta_end, beta_start + (beta_end - beta_start) * (current_step / num_episodes))

class ReplayBuffer:
    """
    for DQN (off-policy RL), big buffer of experience
    you don't update weights of the NN as you run
    through the environment, instead you save
    your experience of the environment to this ReplayBuffer
    It has a max-size to fit in certain examples
    """

    def __init__(self, buffer_limit):
        self.buffer = deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            # a tuple that tells us what the state was
            # at a particular point in time
            # we store the current state, the action we chose,
            # the state we ended up in, and whether finished or not
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        # converting the list to a single numpy.ndarray with numpy.array()
        # before converting to a tensor
        s_lst = torch.stack(s_lst)
        a_lst = np.array(a_lst)
        r_lst = np.array(r_lst)
        s_prime_lst = torch.stack(s_prime_lst)
        done_mask_lst = np.array(done_mask_lst)
        return (
            torch.tensor(s_lst, device=device, dtype=torch.float),
            torch.tensor(a_lst, device=device, dtype=torch.int64),
            torch.tensor(r_lst, device=device),
            torch.tensor(s_prime_lst, device=device, dtype=torch.float),
            torch.tensor(done_mask_lst, device=device),
        )

    def size(self):
        return len(self.buffer)

    def save(self, save_path):
        """save in .pkl file"""
        with open(save_path, "wb") as handle:
            pickle.dump(self.buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file_path):
        """load a .pkl file"""
        with open(file_path, "rb") as handle:
            self.buffer = pickle.load(handle)


class PrioritizedReplayBuffer:
    def __init__(self, buffer_limit, alpha=0.6):
        self.buffer = deque(maxlen=buffer_limit)
        self.priorities = deque(maxlen=buffer_limit)
        self.alpha = alpha  #how much prioritize is used

    def put(self, transition):
        priority = max(self.priorities) if self.priorities else 1.0
        priority =  priority ** self.alpha
        self.buffer.append(transition)
        self.priorities.append(float(priority))

    def sample(self, n, beta=0.4):
        # print("self.priorities, ", type(self.priorities))
        scaled_priorities = np.array(self.priorities, dtype=float).flatten()

        # print("scaled_priorities", type(scaled_priorities), scaled_priorities.shape)
        sampling_probs = scaled_priorities / sum(scaled_priorities)

        # print("sampling_probs", sampling_probs, type(sampling_probs), sampling_probs.shape)
        indices = np.random.choice(range(len(self.buffer)), size=n, p=sampling_probs)

        # print("indices", indices, type(indices), indices.shape)
        mini_batch = [self.buffer[idx] for idx in indices]
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        for transition in mini_batch:

            s, a, r, s_prime, done_mask = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        # Convert lists to numpy arrays
        s_lst = torch.stack(s_lst)
        a_lst = np.array(a_lst)
        r_lst = np.array(r_lst)
        s_prime_lst = torch.stack(s_prime_lst)
        done_mask_lst = np.array(done_mask_lst)

        # print("sampling_probs[indices]" , sampling_probs[indices])
        # print("len(self.buffer) * sampling_probs[indices]", len(self.buffer) * sampling_probs[indices])
        # print("beta", beta)
        #importance sampling weights - important for using in replay buffer - https://datascience.stackexchange.com/questions/32873/prioritized-replay-what-does-importance-sampling-really-do
        weights = (len(self.buffer) * sampling_probs[indices]) ** (-beta)
        weights /= weights.max()

        return (
            torch.tensor(s_lst, device=device, dtype=torch.float),
            torch.tensor(a_lst, device=device,dtype=torch.int64),
            torch.tensor(r_lst, device=device),
            torch.tensor(s_prime_lst, device=device, dtype=torch.float),
            torch.tensor(done_mask_lst, device=device),
            indices,
            torch.tensor(weights,  device=device, dtype=torch.float)
        )


    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            priority = (abs(error) + 1e-5) ** self.alpha
            self.priorities[idx] = priority

    def size(self):
        return len(self.buffer)

    def save(self, save_path):
        """Save in .pkl file"""
        with open(save_path, "wb") as handle:
            pickle.dump(self.buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file_path):
        """Load a .pkl file"""
        with open(file_path, "rb") as handle:
            self.buffer = pickle.load(handle)

class EfficientReplayBuffer: #https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/rl/dqn/replay_buffer.py

    def __init__(self, capacity, alpha=0.6):
        """
        ### Initialize
        """
        # We use a power of $2$ for capacity because it simplifies the code and debugging
        self.capacity = capacity
        # $\alpha$
        self.alpha = alpha

        # Maintain segment binary trees to take sum and find minimum over a range
        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]

        # Current max priority, $p$, to be assigned to new transitions
        self.max_priority = 1.

        # Arrays for buffer
        self.data = {
            'obs': [],
            'action': np.zeros(shape=capacity, dtype=np.int32),
            'reward': np.zeros(shape=capacity, dtype=np.float32),
            'next_obs': [],
            'done': np.zeros(shape=capacity, dtype=np.bool)
        }
        # We use cyclic buffers to store data, and `next_idx` keeps the index of the next empty
        # slot
        self.next_idx = 0

        # Size of the buffer
        self.buffer_size = 0

    def put(self, transition):
        obs, action, reward, next_obs, done = transition
        """
        ### Add sample to queue
        """

        # Get next available slot
        idx = self.next_idx

        # store in the queue
        if len(self.data['obs']) < self.capacity:
            # Expand storage if not full
            self.data['obs'].append(obs)
            self.data['next_obs'].append(next_obs)
        else:
            # Replace old entries cyclically
            self.data['obs'][self.next_idx] = obs
            self.data['next_obs'][self.next_idx] = next_obs

        self.data['action'][idx] = action
        self.data['reward'][idx] = reward
        self.data['next_obs'][idx] = next_obs

        # Increment next available slot
        self.next_idx = (idx + 1) % self.capacity
        # Calculate the size
        self.buffer_size = min(self.capacity, self.buffer_size + 1)

        # $p_i^\alpha$, new samples get `max_priority`
        priority_alpha = self.max_priority ** self.alpha
        # Update the two segment trees for sum and minimum
        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)

    def _set_priority_min(self, idx, priority_alpha):
        """
        #### Set priority in binary segment tree for minimum
        """

        # Leaf of the binary tree
        idx += self.capacity
        self.priority_min[idx] = priority_alpha

        # Update tree, by traversing along ancestors.
        # Continue until the root of the tree.
        while idx >= 2:
            # Get the index of the parent node
            idx //= 2
            # Value of the parent node is the minimum of it's two children
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])

    def _set_priority_sum(self, idx, priority):
        """
        #### Set priority in binary segment tree for sum
        """

        # Leaf of the binary tree
        idx += self.capacity
        # Set the priority at the leaf
        self.priority_sum[idx] = priority

        # Update tree, by traversing along ancestors.
        # Continue until the root of the tree.
        while idx >= 2:
            # Get the index of the parent node
            idx //= 2
            # Value of the parent node is the sum of it's two children
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]

    def _sum(self):
        """
        #### $\sum_k p_k^\alpha$
        """

        # The root node keeps the sum of all values
        return self.priority_sum[1]

    def _min(self):
        """
        #### $\min_k p_k^\alpha$
        """

        # The root node keeps the minimum of all values
        return self.priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum):
        """
        #### Find largest $i$ such that $\sum_{k=1}^{i} p_k^\alpha  \le P$
        """

        # Start from the root
        idx = 1
        while idx < self.capacity:
            # If the sum of the left branch is higher than required sum
            if self.priority_sum[idx * 2] > prefix_sum:
                # Go to left branch of the tree
                idx = 2 * idx
            else:
                # Otherwise go to right branch and reduce the sum of left
                #  branch from required sum
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1

        # We are at the leaf node. Subtract the capacity by the index in the tree
        # to get the index of actual value
        return idx - self.capacity

    def sample(self, batch_size, beta,  max_seq_length=100):
        """
        ### Sample from buffer
        """

        # Initialize samples
        samples = {
            'weights': np.zeros(shape=batch_size, dtype=np.float32),
            'indexes': np.zeros(shape=batch_size, dtype=np.int32)
        }

        # Get sample indexes
        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            samples['indexes'][i] = idx

        # $\min_i P(i) = \frac{\min_i p_i^\alpha}{\sum_k p_k^\alpha}$
        prob_min = self._min() / self._sum()
        # $\max_i w_i = \bigg(\frac{1}{N} \frac{1}{\min_i P(i)}\bigg)^\beta$
        max_weight = (prob_min * self.buffer_size) ** (-beta)

        for i in range(batch_size):
            idx = samples['indexes'][i]
            # $P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$
            prob = self.priority_sum[idx + self.capacity] / self._sum()
            # $w_i = \bigg(\frac{1}{N} \frac{1}{P(i)}\bigg)^\beta$
            weight = (prob * self.buffer_size) ** (-beta)
            # Normalize by $\frac{1}{\max_i w_i}$,
            #  which also cancels off the $\frac{1}{N}$ term
            samples['weights'][i] = weight / max_weight



        # Get sampled experiences
        obs_batch = [self.data['obs'][idx] for idx in samples['indexes']]
        next_obs_batch = [self.data['next_obs'][idx] for idx in samples['indexes']]
        action_batch = self.data['action'][samples['indexes']]
        reward_batch = self.data['reward'][samples['indexes']]
        done_batch = self.data['done'][samples['indexes']]

        # Pad sequences for batch processing
        obs_padded = self._pad_sequences(obs_batch, max_seq_length)
        next_obs_padded = self._pad_sequences(next_obs_batch, max_seq_length)

        return (
            torch.tensor(obs_padded, device=device, dtype=torch.float32),
            torch.tensor(action_batch, device=device, dtype=torch.int64),
            torch.tensor(reward_batch, device=device, dtype=torch.float32),
            torch.tensor(next_obs_padded, device=device, dtype=torch.float32),
            torch.tensor(done_batch, device=device, dtype=torch.bool),
            torch.tensor(samples['weights'], device=device, dtype=torch.float32),
            np.array(samples['indexes'])
        )

    def _pad_sequences(self, sequences, max_len, pad_value=-1):
        """
        Pads sequences to `max_len` using `pad_value` (-1) for ignored timesteps.
        """
        padded_batch = []

        for seq in sequences:
            seq_len = len(seq)
            seq = np.array(seq, dtype=np.float32)
            if seq_len < max_len:
                padding = np.full((max_len - seq_len, seq.shape[1]), pad_value, dtype=np.float32)
                padded_seq = np.vstack([seq, padding])
            else:
                padded_seq = seq
            padded_batch.append(padded_seq)

        return np.array(padded_batch,  dtype=np.float32)

    def update_priorities(self, indexes, priorities):
        """
        ### Update priorities
        """
        # print("indexes, priorities", indexes.shape, priorities.shape)
        for idx, priority in zip(indexes, priorities):
            # Set current max priority
            self.max_priority = max(self.max_priority, priority)

            # Calculate $p_i^\alpha$
            priority_alpha = (np.abs(priority) + 1e-5) ** self.alpha
            # Update the trees
            self._set_priority_min(int(idx), priority_alpha)
            self._set_priority_sum(int(idx), priority_alpha)

    def is_full(self):
        """
        ### Whether the buffer is full
        """
        return self.capacity == self.buffer_size

    def size(self):
        return self.buffer_size

    def save(self, save_path):
        """save in .pkl file"""
        with open(save_path, "wb") as handle:
            pickle.dump(self.buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file_path):
        """load a .pkl file"""
        with open(file_path, "rb") as handle:
            self.buffer = pickle.load(handle)



if args.buffer == 'random':
    memory = ReplayBuffer(buffer_limit)
else:
    memory = EfficientReplayBuffer(buffer_limit)

# time the experiment
start_time = time()


def ExponentialDecay(
    episode,
    num_episodes,
    min_exploration_rate,
    max_exploration_rate,
    exploration_decay_rate=5,
    start_decay=0,
):
    decay_duration = num_episodes - start_decay
    exploration_rate = max_exploration_rate
    if episode > start_decay:
        exploration_rate = min_exploration_rate + (
            max_exploration_rate - min_exploration_rate
        ) * np.exp(-exploration_decay_rate * (episode - start_decay) / decay_duration)
    return exploration_rate


def train(q, q_target, memory, optimizer, n_episode, num_episodes, agent_choice = 'dqn', buffer='random'):
    """
    core algorithm of Deep Q-learning

    do this training once per evaluation of the environment
    run evaluation once and train X times
    """
    for i in range(train_times):
        if buffer == 'random':
            s, a, r, s_prime, done_mask = memory.sample(batch_size)

        else:
            beta = get_beta(n_episode, num_episodes)
            s, a, r, s_prime, done_mask, indices, weights = memory.sample(batch_size, beta)
            # print("sizes:", s.shape, a.shape, r.shape, s_prime.shape, done_mask.shape)
            a = a.unsqueeze(1)
            r = r.unsqueeze(1)
            done_mask = done_mask.unsqueeze(1)
        q_out = q(s)
        q_a = q_out.gather(1, a)

        if agent_choice == "ddqn":
            max_action = q(s_prime).max(1)[1].unsqueeze(1)
            # Get Q-value for best action
            max_q_prime = q_target(s_prime).gather(1, max_action)
        else:
            max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)

        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        if buffer != 'random':
            errors = (target - q_a).detach().cpu().numpy()  # TD error
            memory.update_priorities(indices, errors)
            loss = (loss * torch.tensor(weights, device=device, dtype=torch.float)).mean()


        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        if i == 0:
            # Log loss onto Tensorboard
            writer.add_scalar("Q-Network Loss", loss, n_episode)
            writer.add_scalar("Q Mean Values/Q_Network", q(s).mean().item(), n_episode)
            writer.add_scalar("Q Mean Values/Q_Target_Network", q_target(s).mean().item(), n_episode)

######training loop########@
for n_episode in tqdm(range(num_episodes)):
    epsilon = ExponentialDecay(
        n_episode,
        num_episodes,
        min_exploration_rate,
        max_exploration_rate,
        exploration_decay_rate=exploration_decay_rate,
        start_decay=start_decay,
    )

    if use_curriculum:
        if np.random.rand() < revisit_probability and current_seq_idx > 0:
            sampled_idx = np.random.randint(0, current_seq_idx + 1)
        else:
            sampled_idx = current_seq_idx
    else:
        #randomly pick any sequence
        sampled_idx = np.random.randint(0, len(seq_list))

    # reset the environment
    # Initialize the environment and state
    current_seq, opt_reward = seq_list[sampled_idx]
    s = env.reset(options={"new_seq": current_seq, "maximal": opt_reward})
    max_steps_per_episode = len(current_seq)
    s = one_hot_state(s[0][0], s[0][1])

    done = False
    score = 0.0


    for step in range(max_steps_per_episode):
        # unsqueeze(0) adds a dimension at 0th for batch=1
        # i.e. adds a batch dimension
        a = q.sample_action(s.float().unsqueeze(0), epsilon)

        # take the step and get the returned observation s_prime
        (s_prime, r, terminated, truncated, info) = env.step(a)
        # update a to the actual action taken, since the environment might take a different one
        # to enforce the first turn left constraint and avoid collisions
        a = info["actions"][-1]
        done = terminated or truncated

        # Only keep first turn of Left
        # internal 3actionStateEnv self.last_action updated
        # a = env.last_action
        # print("internal 3actionStateEnv last_action = ", a)

        """
        for NN:
            "one-hot" --> return the one-hot version of the quaternary tuple
        """
        s_prime = one_hot_state(s_prime[0], s_prime[1])
        # state_E_col, step_E_col)

        # NOTE: done_mask is for when you get the end of a run,
        # then is no future reward, so we mask it with done_mask
        done_mask = 0.0 if done else 1.0

        memory.put((s, a, r, s_prime, done_mask))
        s = s_prime

        # Add new reward
        # NOTE: Sep15 update partial_reward to be delta instead of progress*curr_reward
        # NOTE: Sep19 update reward to be a tuple, and reward is 0 until done
        score += r

        # check if the last action ended the episode
        if done:
            break

    recent_rewards.append(score)
    # eventually if memory is big enough, we start running the training loop
    # start training after 2000 (for eg) can get a wider distribution
    if memory.size() > mem_start_train:
        train(q, q_target, memory, optimizer, n_episode,num_episodes, agent_choice,buffer)

    # Update the target network, copying all weights and biases in DQN
    if n_episode % TARGET_UPDATE == 0:
        q_target.load_state_dict(q.state_dict())

    # Check if its time to move to the next sequence
    if use_curriculum and n_episode > moving_avg_window and current_seq_idx < len(seq_list) - 1:
        avg_reward = np.mean(recent_rewards)
        if avg_reward >= progress_threshold:
            print(f"Moving to next sequence {current_seq_idx + 1} (Avg Reward: {avg_reward:.2f})")
            current_seq_idx += 1
            recent_rewards.clear()


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
            "Episode {}, score: {:.1f}, epsilon: {:.2f}, reward_max: {}".format(
                n_episode,
                score,
                epsilon,
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

# Save the pytorch model
# Saving & Loading Model for Inference
# Save/Load state_dict (Recommended)
torch.save(q.state_dict(), f"{save_path}{config_str}-state_dict.pth")

env.close()
