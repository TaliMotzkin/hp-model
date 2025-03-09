from typing import Any, Mapping, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from skrl import config
from skrl.agents.torch.dqn import DQN


class DQN_Huber(DQN):
    def __init__(
        self, models, memory, observation_space, action_space, device, cfg, delta=1.0
    ):
        super().__init__(models, memory, observation_space, action_space, device, cfg)
        self.delta = delta  # Huber loss delta parameter

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
            ) = self.memory.sample(
                names=self.tensors_names, batch_size=self._batch_size
            )[
                0
            ]

            with torch.autocast(
                device_type=self._device_type, enabled=self._mixed_precision
            ):

                sampled_states = self._state_preprocessor(sampled_states, train=True)
                sampled_next_states = self._state_preprocessor(
                    sampled_next_states, train=True
                )

                # compute target values
                with torch.no_grad():
                    next_q_values, _, _ = self.target_q_network.act(
                        {"states": sampled_next_states}, role="target_q_network"
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
                    self.q_network.act({"states": sampled_states}, role="q_network")[0],
                    dim=1,
                    index=sampled_actions.long(),
                )

                # Use Huber loss instead of MSE loss
                q_network_loss = F.smooth_l1_loss(
                    q_values, target_values, beta=self.delta
                )

            # optimize Q-network
            self.optimizer.zero_grad()
            self.scaler.scale(q_network_loss).backward()

            if config.torch.is_distributed:
                self.q_network.reduce_parameters()

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # update target network
            if not timestep % self._target_update_interval:
                self.target_q_network.update_parameters(
                    self.q_network, polyak=self._polyak
                )

            # update learning rate
            if self._learning_rate_scheduler:
                self.scheduler.step()

            # record data
            self.track_data("Loss / Q-network loss", q_network_loss.item())

            self.track_data("Target / Target (max)", torch.max(target_values).item())
            self.track_data("Target / Target (min)", torch.min(target_values).item())
            self.track_data("Target / Target (mean)", torch.mean(target_values).item())

            if self._learning_rate_scheduler:
                self.track_data(
                    "Learning / Learning rate", self.scheduler.get_last_lr()[0]
                )
