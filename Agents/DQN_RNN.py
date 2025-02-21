from typing import Any, Mapping, Optional, Tuple, Union

import copy
import math
import gymnasium
from packaging import version

import torch
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model


CUSTOM_DEFAULT_CONFIG = {
    "gradient_steps": 1,  # gradient steps
    "batch_size": 10,  # training batch size

    "discount_factor": 0.99,  # discount factor (gamma)
    "polyak": 0.005,  # soft update hyperparameter (tau)

    "learning_rate": 1e-3,  # learning rate
    "learning_rate_scheduler": None,  # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},  # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,  # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},  # state preprocessor's kwargs (e.g. {"size": env.observation_space})

    "random_timesteps": 0,  # random exploration steps
    "learning_starts": 0,  # learning starts after this many steps

    "update_interval": 1,  # agent update interval
    "target_update_interval": 10,  # target network update interval

    "exploration": {
        "initial_epsilon": 1.0,  # initial epsilon for epsilon-greedy exploration
        "final_epsilon": 0.05,  # final epsilon for epsilon-greedy exploration
        "timesteps": 1000,  # timesteps for epsilon-greedy decay
    },

    "rewards_shaper": None,  # rewards shaping function: Callable(reward, timestep, timesteps) -> reward

    "mixed_precision": False,  # enable automatic mixed precision for higher performance

    "experiment": {
        "directory": "",  # experiment's parent directory
        "experiment_name": "",  # experiment name
        "write_interval": "auto",  # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",  # interval for checkpoints (timesteps)
        "store_separately": False,  # whether to store checkpoints separately

        "wandb": False,  # whether to use Weights & Biases
        "wandb_kwargs": {}  # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}


class CUSTOM(Agent):
    def __init__(self,
                 models: Mapping[str, Model],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Custom agent

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: None)
        :type observation_space: int, tuple or list of integers, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: None)
        :type action_space: int, tuple or list of integers, gymnasium.Space or None, optional
        :param device: Device on which a torch tensor is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda:0"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict
        """
        _cfg = copy.deepcopy(CUSTOM_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(models=models,
                         memory=memory,
                         observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         cfg=_cfg)
        # =======================================================================
        # models
        self.q_network = self.models.get("q_network", None)
        self.target_q_network = self.models.get("target_q_network", None)

        # checkpoint models
        self.checkpoint_modules["q_network"] = self.q_network
        self.checkpoint_modules["target_q_network"] = self.target_q_network

        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.q_network is not None:
                self.q_network.broadcast_parameters()

        if self.target_q_network is not None:
            # freeze target networks with respect to optimizers (update via .update_parameters())
            self.target_q_network.freeze_parameters(True)

            # update target networks (hard update)
            self.target_q_network.update_parameters(self.q_network, polyak=1)


        self._gradient_steps = self.cfg["gradient_steps"]
        self._batch_size = self.cfg["batch_size"]

        self._discount_factor = self.cfg["discount_factor"]
        self._polyak = self.cfg["polyak"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._update_interval = self.cfg["update_interval"]
        self._target_update_interval = self.cfg["target_update_interval"]

        self._exploration_initial_epsilon = self.cfg["exploration"]["initial_epsilon"]
        self._exploration_final_epsilon = self.cfg["exploration"]["final_epsilon"]
        self._exploration_timesteps = self.cfg["exploration"]["timesteps"]

        self._rewards_shaper = self.cfg["rewards_shaper"]

        self._mixed_precision = self.cfg["mixed_precision"]

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # set up optimizer and learning rate scheduler
        if self.q_network is not None:
            self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self._learning_rate)
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(
                    self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )

            self.checkpoint_modules["optimizer"] = self.optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor #input


    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)

            self.tensors_names = ["states", "actions", "rewards", "next_states", "terminated", "truncated"]

        # RNN specifications
        self._rnn = False  # flag to indicate whether RNN is available
        self._rnn_tensors_names = []  # used for sampling during training
        self._rnn_final_states = {"q_network": []}
        self._rnn_initial_states = {"q_network": []}
        #get spesification {} or  {'rnn': {'sizes': [(1, 4, 64), (1, 4, 64)]}}
        self._rnn_sequence_length = self.q_network.get_specification().get("rnn", {}).get("sequence_length", 1)

        # policy
        for i, size in enumerate(self.q_network.get_specification().get("rnn", {}).get("sizes", [])):
            self._rnn = True
            # create tensors in memory
            if self.memory is not None:
                self.memory.create_tensor(
                    name=f"rnn_q_network_{i}", size=(size[0], size[2]), dtype=torch.float32, keep_dimensions=True
                )
                self._rnn_tensors_names.append(f"rnn_q_network_{i}")
            # default RNN states
            self._rnn_initial_states["q_network"].append(torch.zeros(size, dtype=torch.float32, device=self.device))


    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """

        rnn = {"rnn": self._rnn_initial_states["q_network"]} if self._rnn else {} #intialize rnn

        print("rnn act ",rnn)
        if not self._exploration_timesteps:#decay period for exploration strategy
            print("rtuen by exploration timesteps")
            values, _, outputs =self.q_network.act({"states": self._state_preprocessor(states), **rnn}, role="q_network")
            actions = torch.argmax(values,dim=1, keepdim=True)
            #todo check whether we need to do argmax over outputs?
            if self._rnn:
                self._rnn_final_states["q_network"] = outputs.get("rnn", [])
            return (
                actions,
                None,
                outputs,
            )

        # sample random actions
        # actions, _, outputs = self.q_network.random_act({"states": self._state_preprocessor(states), **rnn}, role="q_network")
        #
        # if timestep < self._random_timesteps:
        #     print("rtuen by  timesteps")
        #     return actions, None, outputs


        # print("actions random ", actions)
        # sample actions with epsilon-greedy policy
        epsilon = self._exploration_final_epsilon + (
                self._exploration_initial_epsilon - self._exploration_final_epsilon
        ) * math.exp(-1.0 * timestep / self._exploration_timesteps)

        indexes = (torch.rand(states.shape[0], device=self.device) >= epsilon).nonzero().view(-1)
        print("indexes.numel()", indexes.numel(), indexes)

        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            values, _, outputs = self.q_network.act({"states": states[indexes], **rnn}, role="q_network")
        if indexes.numel():
                actions = torch.empty(states.shape[0], device=self.device)
                actions[indexes] = torch.argmax(values, dim=1, keepdim=True)
                print("actions ",actions)
        else:
            num_actions = self.q_network.action_space.n  # Assuming a discrete action space with 'n' possible actions
            actions = torch.randint(low=0, high=num_actions, size=(states.shape[0],), device=self.device)
            print("Random actions", actions)

        if self._rnn:
            self._rnn_final_states["q_network"] = outputs.get("rnn", [])

        # record epsilon
        self.track_data("Exploration / Exploration epsilon", epsilon)

        print("rtuen normal")
        return actions, None, outputs

    def record_transition(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            next_states: torch.Tensor,
            terminated: torch.Tensor,
            truncated: torch.Tensor,
            infos: Any,
            timestep: int,
            timesteps: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        if self.memory is not None:
            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # package RNN states
            rnn_states = {}
            if self._rnn:
                rnn_states.update(
                    {f"rnn_q_network_{i}": s.transpose(0, 1) for i, s in enumerate(self._rnn_initial_states["q_network"])}
                )

            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                **rnn_states,
            )
            for memory in self.secondary_memories:
                memory.add_samples(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    **rnn_states,
                )
        # update RNN states
        if self._rnn:
            # reset states if the episodes have ended
            finished_episodes = (terminated | truncated).nonzero(as_tuple=False)
            if finished_episodes.numel():
                for rnn_state in self._rnn_final_states["q_network"]:
                    rnn_state[:, finished_episodes[:, 0]] = 0

            self._rnn_initial_states = self._rnn_final_states

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if timestep >= self._learning_starts and not timestep % self._update_interval:
            self._update(timestep, timesteps)

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        # gradient steps
        for gradient_step in range(self._gradient_steps):

            # sample a batch from memory
            (
                sampled_states,
                sampled_actions,
                sampled_rewards,
                sampled_next_states,
                sampled_terminated,
                sampled_truncated,
            ) = self.memory.sample(names=self.tensors_names, batch_size=self._batch_size,  sequence_length=self._rnn_sequence_length)[0]

            rnn_q_network = {}
            if self._rnn:
                sampled_rnn = self.memory.sample_by_index(
                    names=self._rnn_tensors_names, indexes=self.memory.get_sampling_indexes()
                )[0]
                rnn_q_network = {
                    "rnn": [s.transpose(0, 1) for s in sampled_rnn],
                    "terminated": sampled_terminated | sampled_truncated,
                }

            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):

                sampled_states = self._state_preprocessor(sampled_states, train=True)
                sampled_next_states = self._state_preprocessor(sampled_next_states, train=True)

                # compute target values
                with torch.no_grad():
                    next_q_values, _, _ = self.target_q_network.act(
                        {"states": sampled_next_states,  **rnn_q_network}, role="target_q_network"
                    )

                    target_q_values = torch.max(next_q_values, dim=-1, keepdim=True)[0]
                    target_values = (
                            sampled_rewards
                            + self._discount_factor
                            * (sampled_terminated | sampled_truncated).logical_not()
                            * target_q_values
                    )

                # compute Q-network loss
                q_values = torch.gather(
                    self.q_network.act({"states": sampled_states,  **rnn_q_network}, role="q_network")[0],
                    dim=1,
                    index=sampled_actions.long(),
                )

                q_network_loss = F.mse_loss(q_values, target_values)

            # optimize Q-network
            self.optimizer.zero_grad()
            self.scaler.scale(q_network_loss).backward()

            if config.torch.is_distributed:
                self.q_network.reduce_parameters()

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # update target network
            if not timestep % self._target_update_interval:
                self.target_q_network.update_parameters(self.q_network, polyak=self._polyak)

            # update learning rate
            if self._learning_rate_scheduler:
                self.scheduler.step()

            # record data
            self.track_data("Loss / Q-network loss", q_network_loss.item())

            self.track_data("Target / Target (max)", torch.max(target_values).item())
            self.track_data("Target / Target (min)", torch.min(target_values).item())
            self.track_data("Target / Target (mean)", torch.mean(target_values).item())

            if self._learning_rate_scheduler:
                self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
