import warnings
# from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch as th
from gym import spaces
# import numba

# from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffer_base import BaseBuffer

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "cpu",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1,
    ):

        super(RolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
        self.pos_value = 0
        self.batch_size = 0
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:

        self.observations = np.zeros((self.buffer_size,) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size,), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size,), dtype=np.float32)
        self.values = np.zeros((self.buffer_size,), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size,), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size,), dtype=np.float32)
        self.generator_ready = False
        self.pos_value = 0
        self.batch_size = 0
        super(RolloutBuffer, self).reset()

    # @staticmethod
    # @numba.njit(cache=True)
    # def _numba_advantage_return(last_values: th.Tensor, values, episode_starts, dones: np.ndarray, rewards, gamma,
    #                             buffer_size, gae_lambda):
    #     advantages = np.zeros_like(rewards)
    #     # last_gae_lam = np.float64(0)
    #     # last_gae_lam_arr = np.full(dones.shape(), 0)
    #     last_gae_lam = np.float64(0)
    #     for step in range(buffer_size - 1, -1, -1):
    #         if step == buffer_size - 1:
    #             next_non_terminal = 1.0 - dones
    #             next_values = last_values
    #             last_gae_lam_arr = np.full(dones.shape, last_gae_lam, dtype=np.float64)
    #         else:
    #             next_non_terminal = 1.0 - episode_starts[step + 1]
    #             next_values = values[step + 1]
    #             last_gae_lam_arr = np.full(dones.shape, last_gae_lam, dtype=np.float64)
    #         # delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
    #         v_target = rewards[step] + gamma * next_values * next_non_terminal
    #         delta = v_target - values[step]
    #         last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam_arr
    #         advantages[step] = last_gae_lam
    #     # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
    #     # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
    #     returns = advantages + values
    #     return returns

    # @numba.jit()
    def compute_returns_and_advantage(self) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.
        """

        last_gae_lam = 0
        # for step in reversed(range(self.buffer_size)):
        for step in range(self.buffer_size - 1, -1, -1):
            next_non_terminal = 1.0 - self.episode_starts[step + 1]
            next_values = self.values[step + 1]

            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            episode_start: np.ndarray,
            value: th.Tensor,
            log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        # old implementation, double copy?
        # self.observations[self.pos] = np.array(obs).copy()
        # self.actions[self.pos] = np.array(action).copy()
        # self.rewards[self.pos] = np.array(reward).copy()
        # self.episode_starts[self.pos] = np.array(episode_start).copy()

        # if value or log_prob is None:
        #     self.observations[self.pos], self.actions[self.pos], self.rewards[self.pos], \
        #     self.episode_starts[self.pos] = obs.copy(), action.copy(), reward.copy(), episode_start.copy()
        # else:
        #     if len(log_prob.shape) == 0:
        #         # Reshape 0-d tensor to avoid error
        #         log_prob = log_prob.reshape(-1, 1)
        #     self.observations[self.pos], self.actions[self.pos], self.rewards[self.pos], self.episode_starts[self.pos], \
        #     self.values[self.pos], self.log_probs[self.pos] = obs.copy(), action.copy(), reward.copy(), \
        #                                                       episode_start.copy(), value.clone().cpu().numpy().flatten(), \
        #                                                       log_prob.clone().cpu().numpy()

        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        self.observations[self.pos], self.actions[self.pos], self.rewards[self.pos], self.episode_starts[self.pos], \
        self.values[self.pos], self.log_probs[self.pos] = obs.copy(), action.copy(), reward.copy(), \
                                                          episode_start.copy(), value.clone().cpu().numpy().flatten(), \
                                                          log_prob.clone().cpu().numpy()

        # self.actions[self.pos] = action.copy()
        # self.rewards[self.pos] = reward.copy()
        # self.episode_starts[self.pos] = episode_start.copy()

        # self.values[self.pos] = value.clone().cpu().numpy().flatten()
        # self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def add_no_val(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            episode_start: np.ndarray,
            log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param log_prob: log probability of the action
            following the current policy.
        """

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        # old implementation, double copy?
        # self.observations[self.pos] = np.array(obs).copy()
        # self.actions[self.pos] = np.array(action).copy()
        # self.rewards[self.pos] = np.array(reward).copy()
        # self.episode_starts[self.pos] = np.array(episode_start).copy()

        # if value or log_prob is None:
        #     self.observations[self.pos], self.actions[self.pos], self.rewards[self.pos], \
        #     self.episode_starts[self.pos] = obs.copy(), action.copy(), reward.copy(), episode_start.copy()
        # else:
        #     if len(log_prob.shape) == 0:
        #         # Reshape 0-d tensor to avoid error
        #         log_prob = log_prob.reshape(-1, 1)
        #     self.observations[self.pos], self.actions[self.pos], self.rewards[self.pos], self.episode_starts[self.pos], \
        #     self.values[self.pos], self.log_probs[self.pos] = obs.copy(), action.copy(), reward.copy(), \
        #                                                       episode_start.copy(), value.clone().cpu().numpy().flatten(), \
        #                                                       log_prob.clone().cpu().numpy()

        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        self.observations[self.pos], self.actions[self.pos], self.rewards[self.pos], self.episode_starts[self.pos], \
        self.log_probs[self.pos] = obs.copy(), action.copy(), reward.copy(), episode_start.copy(), \
                                   log_prob.clone().cpu().numpy()

        # self.actions[self.pos] = action.copy()
        # self.rewards[self.pos] = reward.copy()
        # self.episode_starts[self.pos] = episode_start.copy()

        # self.values[self.pos] = value.clone().cpu().numpy().flatten()
        # self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def set_values(self, values: th.Tensor):
        numpy_vals = values.clone().cpu().numpy().flatten()
        numpy_vals: np.ndarray
        # print(f"numpy_vals shape: {numpy_vals.shape}")
        # print(f"self.vals shape: {self.values.shape}")
        # print(f"numpy_vals swapax: {self.swap_and_unflatten(numpy_vals).shape}")
        self.values[self.pos_value] = self.swap_and_unflatten(numpy_vals)

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def get_non_rand(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        # indices = np.random.permutation(self.buffer_size * self.n_envs)
        indices = np.arange(self.buffer_size)

        self.batch_size = batch_size

        # Return everything, don't create minibatches
        if self.batch_size is None or self.buffer_size < self.batch_size:
            self.batch_size = self.buffer_size

        start_idx = 0
        curr_index = 0

        while curr_index < self.buffer_size:
            self.pos_value = indices[start_idx: start_idx + self.batch_size]
            yield self._get_obs_and_acts(self.pos_value)
            # check if batch is too big for rest of buffer and modify batch size if necessary
            curr_index += self.batch_size
            if start_idx + self.batch_size > self.buffer_size:
                self.batch_size = self.buffer_size - start_idx
                start_idx = self.buffer_size - self.batch_size - 1
            else:
                start_idx += self.batch_size

    def _get_obs_and_acts(self, batch_inds: np.ndarray) -> RolloutBufferSamples:
        self.observations: np.ndarray
        # print(f"buffer obs shape: {self.observations.shape}")  # (num envs, buffer size, obs size)
        # print(f"swapax obs shape: {self.swap_and_flatten(self.observations[batch_inds]).shape}")
        data = (
            self.swap_and_flatten(self.observations[batch_inds]),
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))


class PrioritizedExperienceReplay(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "cpu",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1,
            max_timesteps: int = 100
    ):

        super(PrioritizedExperienceReplay, self).__init__(buffer_size, observation_space, action_space, device,
                                                          n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.advantages, self.timestep = None, None, None, None
        self.returns, self.values, self.log_probs, self.value_est_err = None, None, None, None
        self.indices = None
        self.generator_ready = False
        self.max_timesteps = max_timesteps
        # self.reset()

    def reset(self) -> None:

        self.observations = np.zeros((self.buffer_size,) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros(self.buffer_size, dtype=np.float32)
        self.timestep = np.zeros(self.buffer_size, dtype=np.float32)
        self.value_est_err = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)
        # self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.generator_ready = False
        super(PrioritizedExperienceReplay, self).reset()

    def add_epoch(self, num_epoches: int = 1) -> None:
        self.timestep = self.timestep + num_epoches
        indices = self.timestep < self.max_timesteps
        indices: np.ndarray
        # print(f"add_epoch indices shape: {indices.shape}")
        # indices_obs = np.stack([indices] * self.obs_shape[0], axis=0)
        # indices_actions = np.stack([indices] * self.action_dim, axis=0)
        # print(f"add_epoch indices_obs shape: {indices_obs.shape}")
        # print(f"add_epoch indices_actions shape: {indices_actions.shape}")

        self.timestep = self.timestep[indices]
        self.value_est_err = self.value_est_err[indices]
        self.observations = self.observations[indices]
        self.actions = self.actions[indices]
        self.advantages = self.advantages[indices]
        self.returns = self.returns[indices]
        self.log_probs = self.log_probs[indices]

    def add(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            returns: np.ndarray,
            advantages: np.ndarray,
            # episode_start: np.ndarray,
            value: th.Tensor,
            value_est_err: np.ndarray,
            log_prob: np.ndarray
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param returns:
        :param advantages:
        :param value: estimated value of the current state
            following the current policy.
        :param value_est_err:
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        # print(f"add obs shape: {obs.shape}")
        # print(f"add self obs shape: {self.observations.shape}")
        # print(f"add action shape: {action.shape}")
        # print(f"add self action shape: {self.actions.shape}")

        self.timestep = np.concatenate((self.timestep, np.zeros_like(value_est_err)), axis=0)
        self.observations = np.concatenate((self.observations, obs.copy()), axis=0)
        self.actions = np.concatenate((self.actions, action.copy()), axis=0)
        self.advantages = np.concatenate((self.advantages, advantages.copy()), axis=0)
        self.returns = np.concatenate((self.returns, returns.copy()), axis=0)
        self.value_est_err = np.concatenate((self.value_est_err, value_est_err.copy()), axis=0)
        self.log_probs = np.concatenate((self.log_probs, log_prob.copy()), axis=0)

        ordered_indices = np.argsort(self.value_est_err, axis=0)[::-1]

        self.timestep = self.timestep[ordered_indices]
        self.value_est_err = self.value_est_err[ordered_indices]
        self.observations = self.observations[ordered_indices]
        self.actions = self.actions[ordered_indices]
        self.advantages = self.advantages[ordered_indices]
        self.returns = self.returns[ordered_indices]
        self.log_probs = self.log_probs[ordered_indices]

        self.timestep = self.timestep[:self.buffer_size]
        self.value_est_err = self.value_est_err[:self.buffer_size]
        self.observations = self.observations[:self.buffer_size]
        self.actions = self.actions[:self.buffer_size]
        self.advantages = self.advantages[:self.buffer_size]
        self.returns = self.returns[:self.buffer_size]
        self.log_probs = self.log_probs[:self.buffer_size]

        # self.pos += 1
        # if self.pos == self.buffer_size:
        #     self.full = True

    def update_vals(
            self,
            value_est_err: np.ndarray,
    ) -> None:

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        # if isinstance(self.observation_space, spaces.Discrete):
        #     obs = obs.reshape((self.n_envs,) + self.obs_shape)

        # self.value_est_err[self.indices[0]: self.indices[1]] = value_est_err.copy()
        # self.log_probs[self.indices[0]: self.indices[1]] = log_prob.copy()

        # for single batch only
        # self.value_est_err = value_est_err.copy()
        # self.log_probs = log_prob.copy()

        self.value_est_err[self.indices] = value_est_err.copy()

        # ordered_indices = np.argsort(self.value_est_err, axis=0)[::-1]
        #
        # self.timestep = self.timestep[ordered_indices]
        # self.value_est_err = self.value_est_err[ordered_indices]
        # self.observations = self.observations[ordered_indices]
        # self.actions = self.actions[ordered_indices]
        # self.advantages = self.advantages[ordered_indices]
        # self.returns = self.returns[ordered_indices]
        # self.log_probs = self.log_probs[ordered_indices]
        #
        # self.timestep = self.timestep[:self.buffer_size]
        # self.value_est_err = self.value_est_err[:self.buffer_size]
        # self.observations = self.observations[:self.buffer_size]
        # self.actions = self.actions[:self.buffer_size]
        # self.advantages = self.advantages[:self.buffer_size]
        # self.returns = self.returns[:self.buffer_size]
        # self.log_probs = self.log_probs[:self.buffer_size]

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        # softmax
        exp = np.exp(self.value_est_err)
        probability = exp / np.sum(exp)

        indices = np.random.choice(a=self.buffer_size, size=self.buffer_size, replace=False, p=probability)

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size

        start_idx = 0
        while start_idx < self.buffer_size:
            self.indices = indices[start_idx: start_idx + batch_size]
            # self.indices = [start_idx, start_idx + batch_size]
            yield self._get_samples(self.indices)
            # yield self._get_samples(start_idx, start_idx + batch_size)
            start_idx += batch_size

    # def _get_samples(self, start_ind: int, end_ind: int, env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
        data = (
            # self.observations[start_ind:end_ind],
            # self.actions[start_ind:end_ind],
            # self.values[start_ind:end_ind],
            # self.log_probs[start_ind:end_ind],
            # self.advantages[start_ind:end_ind],
            # self.returns[start_ind:end_ind],
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds],
            self.log_probs[batch_inds],
            self.advantages[batch_inds],
            self.returns[batch_inds],
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
