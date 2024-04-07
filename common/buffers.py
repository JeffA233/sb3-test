import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch as th
from gym import spaces
# import numba

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "cpu",
            n_envs: int = 1,
    ):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    @staticmethod
    def swap_and_flatten_to_tensor(arr: np.ndarray) -> th.Tensor:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return th.from_numpy(arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:]))

    def swap_and_unflatten(self, arr: np.ndarray, second_dim: int = None) -> np.ndarray:
        """
        Undo what swap_and_flatten did
        :param arr:
        :param second_dim:
        :return:
        """
        shape = arr.shape
        if second_dim is None:
            second_dim = shape[0] // self.n_envs
        # if len(shape) < 3:
        #     shape = shape + (1,)
        return arr.reshape((self.n_envs, second_dim, *shape[1:])).swapaxes(1, 0)

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
            self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if isinstance(array, th.Tensor):
            return array.to(device=self.device, non_blocking=True, copy=copy)
        # if copy:
            # return th.tensor(array, device=self.device)
            # return th.from_numpy(array).to(device=self.device, non_blocking=True, copy=copy)
        # return th.as_tensor(array, device=self.device)
        return th.from_numpy(array).to(device=self.device, non_blocking=True, copy=copy)

    @staticmethod
    def _normalize_obs(
            obs: Union[np.ndarray, Dict[str, np.ndarray]],
            env: Optional[VecNormalize] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "cpu",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)

        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape,
                                              dtype=observation_space.dtype)

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> None:

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)
            next_obs = next_obs.reshape((self.n_envs,) + self.obs_shape)

        # Same, for actions
        if isinstance(self.action_space, spaces.Discrete):
            action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = obs.copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = next_obs.copy()
        self.actions[self.pos] = action.copy()
        self.rewards[self.pos] = reward.copy()
        self.dones[self.pos] = done.copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


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
        self.done_indices = None
        self.episodes_done = False
        self.max_mult = 2
        self.reset()

    def reset(self) -> None:

        self.observations = np.zeros((self.buffer_size*self.max_mult, self.n_envs) + self.obs_shape, dtype=np.float16)
        self.actions = np.zeros((self.buffer_size*self.max_mult, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size*self.max_mult, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size*self.max_mult, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size*self.max_mult, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size*self.max_mult, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size*self.max_mult, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size*self.max_mult, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        self.pos_value = 0
        self.batch_size = 0
        # so then we do [:, indices] since first dim is pos
        self.done_indices = np.arange(self.n_envs)
        self.episodes_done = False
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
    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
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

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.cpu().numpy().flatten()

        last_gae_lam = 0
        # for step in reversed(range(self.buffer_size)):
        for step in range(self.buffer_size*self.max_mult - 1, -1, -1):
            if step == self.buffer_size*self.max_mult - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

        # truncate now before we convert to tensors but after we calculate advantages
        self.observations = self.observations[:self.buffer_size]
        self.actions = self.actions[:self.buffer_size]
        self.rewards = self.rewards[:self.buffer_size]
        self.returns = self.returns[:self.buffer_size]
        self.episode_starts = self.episode_starts[:self.buffer_size]
        self.values = self.values[:self.buffer_size]
        self.log_probs = self.log_probs[:self.buffer_size]
        self.advantages = self.advantages[:self.buffer_size]

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

        if self.full:
            removal_indices = np.asarray(episode_start).nonzero()
            self.done_indices = np.delete(self.done_indices, removal_indices)

        if len(self.done_indices) == 0 or self.pos == self.buffer_size*self.max_mult - 1:
            self.episodes_done = True

        self.observations[self.pos][self.done_indices], \
            self.actions[self.pos][self.done_indices], \
            self.rewards[self.pos][self.done_indices], \
            self.episode_starts[self.pos][self.done_indices], \
            self.values[self.pos][self.done_indices], \
            self.log_probs[self.pos][self.done_indices] = \
            obs[self.done_indices].copy(), \
            action[self.done_indices].copy(), \
            reward[self.done_indices].copy(), \
            episode_start[self.done_indices].copy(), \
            value[self.done_indices].clone().cpu().numpy().flatten(), \
            log_prob[self.done_indices].clone().cpu().numpy()

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

        # if self.full:
        #     removal_indices = np.asarray(episode_start[self.done_indices]).nonzero()
        #     if len(removal_indices[0]) != 0:
        #         thing = 0
        #     self.done_indices = np.delete(self.done_indices, removal_indices)

        if len(self.done_indices) == 0 or self.pos == self.buffer_size*self.max_mult - 1:
            self.episodes_done = True
        else:
            self.observations[self.pos], \
                self.actions[self.pos], \
                self.rewards[self.pos], \
                self.episode_starts[self.pos], \
                self.log_probs[self.pos] = \
                obs, \
                action, \
                reward, \
                episode_start, \
                log_prob.cpu().numpy()

        if self.full:
            removal_indices = np.asarray(episode_start[self.done_indices]).nonzero()
            # if len(removal_indices[0]) != 0:
                # thing = 0
            self.done_indices = np.delete(self.done_indices, removal_indices)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def set_values(self, values: th.Tensor):
        numpy_vals = values.cpu().numpy().flatten()
        numpy_vals: np.ndarray
        # print(f"numpy_vals shape: {numpy_vals.shape}")
        # print(f"self.vals shape: {self.values.shape}")
        # print(f"numpy_vals swapax: {self.swap_and_unflatten(numpy_vals).shape}")
        # print(f"pos_value in values: {self.pos_value}")
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
                # self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
                self.__dict__[tensor] = self.swap_and_flatten_to_tensor(self.__dict__[tensor])
                # self.__dict__[tensor] = self.__dict__[tensor].pin_memory()
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def get_minibatch(self, batch_size: Optional[int] = None, minibatch_size: Optional[int] = None) \
            -> (Generator[RolloutBufferSamples, None, None], bool):
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
                # self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
                self.__dict__[tensor] = self.swap_and_flatten_to_tensor(self.__dict__[tensor])
                # self.__dict__[tensor] = self.__dict__[tensor].pin_memory()
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs
        if minibatch_size is None:
            minibatch_size = batch_size
        if batch_size is None or self.buffer_size * self.n_envs < batch_size:
            batch_size = self.buffer_size * self.n_envs
            # print(f"adjusted batch size to be: {batch_size}")

        self.batch_size = batch_size

        start_idx_batch, start_idx, minibatch_size_trunc = 0, 0, 0
        trunc_or_done_batch_bool, minibatch_trunc_or_done = False, False
        # dbg
        # mini_dbg_val, batch_dbg_val = 0, 0
        # -
        # NOTE: this is mostly useless as indexes beyond the end of the array/list are not counted anyways in Python.
        # TL;DR most of this is then not necessary when compared to get() but in the future it may be useful when
        # writing in other languages which require it.
        while True:
            # print(f"batch {batch_dbg_val} {start_idx_batch}")
            while True:
                # print(f"minibatch {mini_dbg_val} {start_idx}")
                if not minibatch_trunc_or_done:
                    first_idx = start_idx_batch + start_idx
                    end_idx = start_idx_batch + start_idx + minibatch_size
                    idxs = indices[first_idx:end_idx]
                else:
                    first_idx = start_idx_batch + start_idx
                    # end_idx = start_idx_batch + start_idx + minibatch_size_trunc
                    end_idx = start_idx_batch + start_idx + minibatch_size
                    idxs = indices[first_idx:end_idx]
                # dbg and error checking
                if len(idxs) < 1 or len(idxs) > minibatch_size:
                    start_idx_print = start_idx_batch + start_idx
                    end_idx_print = start_idx_batch + start_idx + minibatch_size
                    index_array_len = len(idxs)
                    print(f"got irregular index array: {start_idx_print}, {end_idx_print}, {index_array_len}, "
                          f"{minibatch_trunc_or_done}, {trunc_or_done_batch_bool}, check minibatch func in buffer")
                # -
                if self.batch_size == minibatch_size:
                    minibatch_trunc_or_done = True
                yield self._get_samples(idxs), minibatch_trunc_or_done
                if minibatch_trunc_or_done:
                    break
                # we need to do 2* because we're going to do
                # [start_idx + minibatch_size : end_idx + 2*minibatch_size]
                # essentially after this statement
                if start_idx + 2*minibatch_size >= self.batch_size - 1:
                    # minibatch_size_trunc = batch_size - start_idx
                    # start_idx = batch_size - minibatch_size_trunc
                    start_idx += minibatch_size
                    minibatch_trunc_or_done = True
                    # dbg
                    # if start_idx > 1_000_000:
                    # print(f"truncated debug values: {start_idx}, {self.batch_size}, {minibatch_size}")
                    # -
                    
                    # we must break so as not to do another loop with a 0-sized array
                    if start_idx + minibatch_size == self.batch_size - 1:
                        break
                else:
                    start_idx += minibatch_size
                    # in case minibatch perfectly cuts batch size
                    # if (start_idx + minibatch_size) == batch_size:
                    #     minibatch_size_trunc = minibatch_size
                    #     minibatch_trunc_or_done = True
                # dbg
                # mini_dbg_val += 1
                # print(f"debug values: {start_idx}, {self.batch_size} {minibatch_size}")
                # print(f"minibatch {mini_dbg_val} {start_idx}")
                # -
            start_idx = 0
            minibatch_trunc_or_done = False
            # dbg
            # if mini_dbg_val > 7:
            #     print(f"mini_dbg_val was {mini_dbg_val}")
            # mini_dbg_val = 0
            # -
            # minibatch_size_trunc = 0
            # start_idx_batch += batch_size
            if trunc_or_done_batch_bool:
                break
            if start_idx_batch + self.batch_size > self.buffer_size * self.n_envs - 1:
                # trunc_batch_size = self.buffer_size * self.n_envs - start_idx_batch
                # start_idx_batch = self.buffer_size * self.n_envs - trunc_batch_size
                
                # start_idx_batch += self.batch_size
                # trunc_or_done_batch_bool = True
                break
                
                # we must break so as not to do another loop with a 0-sized array
                # if start_idx_batch + self.batch_size == self.buffer_size * self.n_envs - 1:
                #     break
            else:
                start_idx_batch += self.batch_size
                # in case batch perfectly cuts buffer size
                # if start_idx_batch + batch_size >= self.buffer_size * self.n_envs:
                #     trunc_or_done_batch_bool = True
            # dbg
            # batch_dbg_val += 1
            # if batch_dbg_val > 2:
            #     print(f"batch_dbg_val was: {batch_dbg_val}")
            # print(f"batch {batch_dbg_val} {start_idx_batch}")
        if not end_idx >= self.buffer_size * self.n_envs - 1:
            print(f"end_idx was not correct: {end_idx}")
        # -

    def get_non_rand(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        # indices = np.random.permutation(self.buffer_size * self.n_envs)
        actual_buffer_size = self.buffer_size*self.max_mult
        indices = np.arange(actual_buffer_size)

        self.batch_size = batch_size
        
        # Return everything, don't create minibatches
        if self.batch_size is None or actual_buffer_size < self.batch_size:
            self.batch_size = self.buffer_size

        start_idx = 0
        # curr_index = 0
        trunc_batch_bool = False

        # while curr_index < self.buffer_size:
        while True:
            self.pos_value = indices[start_idx: start_idx + self.batch_size]
            yield self._get_obs_and_acts(self.pos_value)
            # check if batch is too big for rest of buffer and modify batch size if necessary
            # curr_index += self.batch_size
            if trunc_batch_bool:
                break
            if start_idx + self.batch_size > actual_buffer_size:
                self.batch_size = actual_buffer_size - start_idx
                start_idx = self.buffer_size - self.batch_size
                trunc_batch_bool = True
            else:
                start_idx += self.batch_size
                if start_idx + self.batch_size == actual_buffer_size:
                    trunc_batch_bool = True

        self.batch_size = batch_size
        # print(f"last buffer value was: {self.values[-1]}")

    def _get_obs_and_acts(self, batch_inds: np.ndarray) -> RolloutBufferSamples:
        self.observations: np.ndarray
        # print(f"buffer obs shape: {self.observations[batch_inds].shape}")  # (num envs, batch size, obs size)
        # print(f"swapax obs shape: {self.swap_and_flatten(self.observations[batch_inds]).shape}")
        # print(f"pos_value in _get_obs_and_acts: {batch_inds}")
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
        copy_bools = (
            False,
            False,
            False,
            False,
            False,
            False,
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data, copy_bools)))


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

        self.observations = th.zeros((self.buffer_size,) + self.obs_shape, dtype=th.float32)
        if isinstance(self.action_space, spaces.discrete.Discrete):
            self.actions = th.zeros(self.buffer_size, dtype=th.float32)
        else:
            self.actions = th.zeros((self.buffer_size, self.action_dim), dtype=th.float32)
        self.timestep = th.zeros(self.buffer_size, dtype=th.float32)
        self.value_est_err = th.zeros(self.buffer_size, dtype=th.float32)
        self.returns = th.zeros(self.buffer_size, dtype=th.float32)
        # self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = th.zeros(self.buffer_size, dtype=th.float32)
        self.log_probs = th.zeros(self.buffer_size, dtype=th.float32)
        self.advantages = th.zeros(self.buffer_size, dtype=th.float32)
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
            obs: th.Tensor,
            action: th.Tensor,
            returns: th.Tensor,
            advantages: th.Tensor,
            # episode_start: np.ndarray,
            value: th.Tensor,
            value_est_err: th.Tensor,
            log_prob: th.Tensor
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

        self.timestep = th.concatenate((self.timestep, th.zeros_like(value_est_err)), dim=0)
        self.observations = th.concatenate((self.observations, obs), dim=0)
        self.actions = th.concatenate((self.actions, action), dim=0)
        self.advantages = th.concatenate((self.advantages, advantages), dim=0)
        self.returns = th.concatenate((self.returns, returns), dim=0)
        self.value_est_err = th.concatenate((self.value_est_err, value_est_err), dim=0)
        self.log_probs = th.concatenate((self.log_probs, log_prob), dim=0)

        ordered_indices = th.argsort(self.value_est_err, dim=0, descending=True)

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
            value_est_err: th.Tensor,
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

        self.value_est_err[self.indices] = value_est_err

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
        if batch_size == 0:
            return None
        # softmax
        # clip because exp blows up in size if not careful?
        exp = th.exp(th.clip(self.value_est_err, -10, 10))
        probability = exp / th.sum(exp)
        probability: th.Tensor

        # indices = np.random.choice(a=probability.size, size=probability.size, replace=False, p=probability)
        th_range = th.arange(0, probability.size(dim=0), dtype=th.int32)
        selection = probability.multinomial(num_samples=probability.size(dim=0))
        indices = th_range[selection]

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size

        start_idx = 0
        while start_idx < self.buffer_size:
            self.indices = indices[start_idx: start_idx + batch_size]
            self.indices: th.Tensor
            # self.indices = [start_idx, start_idx + batch_size]
            yield self._get_samples(self.indices)
            # yield self._get_samples(start_idx, start_idx + batch_size)
            start_idx += batch_size

    # def _get_samples(self, start_ind: int, end_ind: int, env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
    def _get_samples(self, batch_inds: th.Tensor, env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
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


class DictReplayBuffer(ReplayBuffer):
    """
    Dict Replay buffer used in off-policy algorithms like SAC/TD3.
    Extends the ReplayBuffer to use dictionary observations

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "cpu",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        assert optimize_memory_usage is False, "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = {
            key: np.zeros((self.buffer_size, self.n_envs) + _obs_shape, dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        self.next_observations = {
            key: np.zeros((self.buffer_size, self.n_envs) + _obs_shape, dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            obs_nbytes = 0
            for _, obs in self.observations.items():
                obs_nbytes += obs.nbytes

            total_memory_usage = obs_nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            if self.next_observations is not None:
                next_obs_nbytes = 0
                for _, obs in self.observations.items():
                    next_obs_nbytes += obs.nbytes
                total_memory_usage += next_obs_nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
            self,
            obs: Dict[str, np.ndarray],
            next_obs: Dict[str, np.ndarray],
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> None:
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = np.array(obs[key])

        for key in self.next_observations.keys():
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.next_observations[key][self.pos] = np.array(next_obs[key]).copy()

        # Same reshape, for actions
        if isinstance(self.action_space, spaces.Discrete):
            action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        return super(ReplayBuffer, self).sample(batch_size=batch_size, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()})
        next_obs_ = self._normalize_obs(
            {key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()})

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(
                self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(
                -1, 1
            ),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
        )


class DictRolloutBuffer(RolloutBuffer):
    """
    Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
    Extends the RolloutBuffer to use dictionary observations

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
        Equivalent to Monte-Carlo advantage estimate when set to 1.
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

        assert isinstance(self.obs_shape, dict), "DictRolloutBuffer must be used with Dict obs space only"

        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        assert isinstance(self.obs_shape, dict), "DictRolloutBuffer must be used with Dict obs space only"
        self.observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            self.observations[key] = np.zeros((self.buffer_size, self.n_envs) + obs_input_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def add(
            self,
            obs: Dict[str, np.ndarray],
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
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            obs_ = np.array(obs[key]).copy()
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = obs_

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[DictRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = ["actions", "values", "log_probs", "advantages", "returns"]

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

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> DictRolloutBufferSamples:

        return DictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
        )
