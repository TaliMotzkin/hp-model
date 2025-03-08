import gymnasium as gym
# import the skrl components to build the RL system
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.model_instantiators.torch import Shape, deterministic_model
from torch.optim.lr_scheduler import CosineAnnealingLR

import envs
from lstm_qnetwork import LSTMQNetwork

# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed


# load and wrap the gymnasium environment.
# note: the environment version may change depending on the gymnasium version
# seq = "HHHPHPHHPPP"
# seq = "PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH"
seq = "PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP"
env = gym.make_vec("HPEnv_v0", num_envs=3, seq=seq)
env = wrap_env(env)

device = env.device


# instantiate a memory as experience replay
memory = RandomMemory(memory_size=50000, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators) using the model instantiator utility.
# DQN requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/dqn.html#models
models = {}
models["q_network"] = LSTMQNetwork(observation_space=env.observation_space,
                                  action_space=env.action_space,
                                  device=device)
models["target_q_network"] = LSTMQNetwork(observation_space=env.observation_space,
                                         action_space=env.action_space,
                                         device=device)

# initialize models' lazy modules
for role, model in models.items():
    model.init_state_dict(role)

# initialize models' parameters (weights and biases)
for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/dqn.html#configuration-and-hyperparameters
cfg = DQN_DEFAULT_CONFIG.copy()
cfg["batch_size"] = 200
cfg["learning_starts"] = 5000
cfg["exploration"]["final_epsilon"] = 0.01
cfg["exploration"]["timesteps"] = 15000
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 1000
cfg["experiment"]["checkpoint_interval"] = 5000
cfg["learning_rate"] = 0.0005
# cfg["learning_rate_scheduler"] = CosineAnnealingLR
# cfg["learning_rate_scheduler_kwargs"] = {'T_max': 100}
cfg["experiment"]["directory"] = "runs/test/"

agent = DQN(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 500000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.train()
