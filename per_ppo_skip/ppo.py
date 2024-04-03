# import warnings
from typing import Any, Dict, Optional, Type, Union, Iterable
# import warnings
from warnings import warn

import numpy as np
import tqdm

# import torch.nn
from gym import spaces
# import numpy as np
from numpy import mean as np_mean, amax, amin, average
from collections import deque
# import torch as th
from torch import Tensor, device, min, clamp, abs, exp, no_grad, mean, any, isnan, logical_or, logical_and
# from torch.nn import Softmax
# from torch.nn import functional as F
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_

_tensor_or_tensors = Union[Tensor, Iterable[Tensor]]

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicySkip
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.buffers import PrioritizedExperienceReplay


class PER_PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param n_steps_per:
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicySkip]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        n_steps_per: int = 2048,
        max_epoches_per: int = 100,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(PER_PPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        assert (
            batch_size > 1
        ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_steps_per = n_steps_per
        self.max_epoches_per = max_epoches_per
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl
        self.epoch_collection_num = 0

        self.use_adaptive_epoches = False
        self.adaptive_clip_fraction_target = 0.1
        # self.previous_clip_fractions = deque([], maxlen=5)
        self.previous_clip_fraction = 0
        self.clip_fraction_avg_weights = [0.65, 0.125, 0.1, 0.075, 0.05]
        self.max_epoches = 40

        self.PERBuffer = PrioritizedExperienceReplay(
            self.n_steps_per,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            max_timesteps=self.max_epoches_per,
        )

        if _init_setup_model:
            self._setup_model()
            self.PERBuffer.reset()

    def _setup_model(self) -> None:
        super(PER_PPO, self)._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        losses = []
        losses_per = []

        continue_training = True

        progress_bar = tqdm.tqdm(desc=f"Training", total=self.n_epochs, leave=False,
                                 smoothing=0.01)

        # train for n_epochs epochs
        approx_kl_divs = []
        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            progress_bar.set_description(f"Training, rollout buffer | epoch")
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # actions: Tensor
                # if any(isnan(actions)).item():
                #     print(f"nan in actions, epoch num: {epoch}")

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages_nonavg = rollout_data.advantages
                advantages = (advantages_nonavg - advantages_nonavg.mean()) / (advantages_nonavg.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                # policy_loss_1 = advantages * ratio
                # policy_loss_2 = advantages * clamp(ratio, 1 - clip_range, 1 + clip_range)
                # policy_loss = -min(policy_loss_1, policy_loss_2).mean()
                adv_clamped = advantages * clamp(ratio, 1 - clip_range, 1 + clip_range)
                adv_ratio = advantages * ratio
                policy_loss = -min(adv_ratio, adv_clamped).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                # clip_fraction = mean((abs(ratio - 1) > clip_range).float()).item()
                clamped_ratio_float = logical_or(ratio > 1 + clip_range, ratio < 1 - clip_range).float()
                partial = logical_and(clamped_ratio_float, adv_ratio > adv_clamped).float()
                clip_fraction = mean(partial).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )

                value_est_err = abs(rollout_data.returns - values_pred)

                if epoch == 0:
                    self.PERBuffer.add(rollout_data.observations.detach().cpu().numpy(),
                                       actions.detach().cpu().numpy(),
                                       rollout_data.returns.detach().cpu().numpy(),
                                       advantages_nonavg.detach().cpu().numpy(),
                                       values.detach().cpu().numpy(),
                                       value_est_err.detach().cpu().numpy(),
                                       log_prob.detach().cpu().numpy())

                # Value loss using the TD(gae_lambda) target
                value_loss = mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -mean(-log_prob)
                else:
                    entropy_loss = -mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                losses.append(loss.item())

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = mean((exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                progress_bar.set_description(f"Training, rollout backwards | epoch")
                self.policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Clip grad norm
                # th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                if self.max_grad_norm != 0:
                    # self.policy: torch.nn.Module = self.policy.to("cpu", non_blocking=True)
                    # self.clip_grad_norm_test(self.policy.parameters(), self.max_grad_norm)
                    clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    # self.policy = self.policy.to(self.device, non_blocking=True)
                self.policy.optimizer.step()

            if self.epoch_collection_num < 5:
                self.epoch_collection_num += 1

            # now do PERBuffer -------------------------------------------------------

            if self.epoch_collection_num == 5:
                progress_bar.set_description(f"Training, PER buffer | epoch")
                if self.PERBuffer.buffer_size < self.batch_size:
                    per_batch = self.PERBuffer.buffer_size
                else:
                    per_batch = None

                for rollout_data in self.PERBuffer.get(per_batch):
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to long
                        actions = rollout_data.actions.long().flatten()

                    # Re-sample the noise matrix because the log_std has changed
                    if self.use_sde:
                        self.policy.reset_noise(self.batch_size)

                    values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                    values: Tensor
                    values = values.flatten()
                    # Normalize advantage
                    advantages = rollout_data.advantages
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # ratio between old and new policy, should be one at the first iteration
                    ratio = exp(log_prob - rollout_data.old_log_prob)

                    # clipped surrogate loss
                    # policy_loss_1 = advantages * ratio
                    # policy_loss_2 = advantages * clamp(ratio, 1 - clip_range, 1 + clip_range)
                    # policy_loss = -min(policy_loss_1, policy_loss_2).mean()
                    adv_clamped = advantages * clamp(ratio, 1 - clip_range, 1 + clip_range)
                    adv_ratio = advantages * ratio
                    policy_loss = -min(adv_ratio, adv_clamped).mean()

                    # Logging
                    pg_losses.append(policy_loss.item())
                    # clip_fraction = mean((abs(ratio - 1) > clip_range).float()).item()
                    clamped_ratio_float = logical_or(ratio > 1 + clip_range, ratio < 1 - clip_range).float()
                    partial = logical_and(clamped_ratio_float, adv_ratio > adv_clamped).float()
                    clip_fraction = mean(partial).item()
                    clip_fractions.append(clip_fraction)

                    if self.clip_range_vf is None:
                        # No clipping
                        values_pred = values
                    else:
                        # Clip the different between old and new value
                        # NOTE: this depends on the reward scaling
                        values_pred = rollout_data.old_values + clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    value_est_err = abs(rollout_data.returns - values_pred)
                    value_est_err_array = value_est_err.detach().clone().cpu().numpy()

                    self.PERBuffer.update_vals(value_est_err_array)

                    # Value loss using the TD(gae_lambda) target
                    value_loss = mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(value_loss.item())

                    # Entropy loss favor exploration
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = -mean(-log_prob)
                    else:
                        entropy_loss = -mean(entropy)

                    entropy_losses.append(entropy_loss.item())

                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                    losses_per.append(loss.item())

                    # Calculate approximate form of reverse KL Divergence for early stopping
                    # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                    # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                    # and Schulman blog: http://joschu.net/blog/kl-approx.html
                    with no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = mean((exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break

                    # Optimization step
                    progress_bar.set_description(f"Training, PER backwards | epoch")
                    self.policy.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    # Clip grad norm
                    # th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    if self.max_grad_norm != 0:
                        # self.policy: torch.nn.Module = self.policy.to("cpu", non_blocking=True)
                        # self.clip_grad_norm_test(self.policy.parameters(), self.max_grad_norm)
                        clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                        # self.policy = self.policy.to(self.device, non_blocking=True)
                    self.policy.optimizer.step()

            if not continue_training:
                break

            progress_bar.update(1)

        try:
            progress_bar.close()
        except:
            pass

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        clip_fraction_mean = np_mean(clip_fractions)
        if self.use_adaptive_epoches:
            self.previous_clip_fraction = clip_fraction_mean

            # to make sure there are no errors pertaining to the weights not being of the same shape
            # while len(self.previous_clip_fractions) < len(self.clip_fraction_avg_weights):
            #     self.previous_clip_fractions.appendleft(clip_fraction_mean)

            # adjustment = self.adaptive_clip_fraction_target / average(self.previous_clip_fractions,
            #                                                           weights=self.clip_fraction_avg_weights)
            adjustment = self.adaptive_clip_fraction_target / self.previous_clip_fraction
            self.n_epochs = int(self.n_epochs * adjustment)

            # keep it within reason just in case
            if self.n_epochs > self.max_epoches:
                self.n_epochs = self.max_epoches
            elif self.n_epochs < 1:
                self.n_epochs = 1

        # Logs
        self.logger.record("train/entropy_loss", np_mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np_mean(pg_losses))
        self.logger.record("train/value_loss", np_mean(value_losses))
        self.logger.record("train/approx_kl", np_mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", clip_fraction_mean)
        self.logger.record("train/loss", loss.item())
        # new
        self.logger.record("train/loss_mean", np_mean(losses))

        if self.epoch_collection_num == 5:
            self.logger.record("train/per_loss_mean", np_mean(losses_per))
            self.logger.record("train/PRE_buffer_size", self.PERBuffer.value_est_err.size)

            self.logger.record("train/PRE_err_mean", np_mean(self.PERBuffer.value_est_err))
            self.logger.record("train/PRE_err_max", amax(self.PERBuffer.value_est_err))
            self.logger.record("train/PRE_err_min", amin(self.PERBuffer.value_est_err))

            self.logger.record("train/PRE_mean_timestep", np_mean(self.PERBuffer.timestep))
            self.logger.record("train/PRE_max_timestep", amax(self.PERBuffer.timestep))
            self.logger.record("train/PRE_min_timestep", amin(self.PERBuffer.timestep))
        #
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        self.PERBuffer.add_epoch(self.n_epochs)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "PER-PPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":

        return super(PER_PPO, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )
