import gymnasium as gym
import os
from models.RNN import RNN_class
from Agents.DQN_RNN import CUSTOM
# import the skrl components to build the RL system
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.model_instantiators.torch import Shape, deterministic_model

from envs import Action


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed
DQN

# load and wrap the gymnasium environment.
# note: the environment version may change depending on the gymnasium version

seq = 'HPHPPHHPHPPHPHHPPHPH'
env = gym.make('HPEnv_v0', seq=seq)
env.num_envs = 1 #currently non vectorized enviroenmnt
env = wrap_env(env)

observations, _ = env.reset()

device = env.device


# instantiate a memory as experience replay
memory = RandomMemory(memory_size=500, num_envs=env.num_envs, device=device, replacement=False)


# instantiate the agent's models (function approximators) using the model instantiator utility.
# DQN requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/dqn.html#models
models = {}
models["q_network"] = RNN_class(observation_space=env.observation_space, action_space=env.action_space, device=env.device,
                          clip_actions=False, num_envs=env.num_envs, num_layers=1, hidden_size=32,sequence_length=1)
models["target_q_network"] = RNN_class(observation_space=env.observation_space, action_space=env.action_space, device=env.device,
                          clip_actions=False, num_envs=env.num_envs, num_layers=1, hidden_size=32,sequence_length=1)


# initialize models' lazy modules
# for role, model in models.items():
#     model.init_state_dict(role)

# initialize models' parameters (weights and biases)
for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/dqn.html#configuration-and-hyperparameters
cfg = DQN_DEFAULT_CONFIG.copy()
cfg["learning_starts"] = 100
cfg["discount_factor"] = 0.98
cfg["exploration"]["final_epsilon"] = 0.04
cfg["exploration"]["timesteps"] = 1500
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 1000
cfg["experiment"]["checkpoint_interval"] = 5000
cfg["experiment"]["directory"] = "runs/hp-model"

agent = DQN(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 50000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.train()
