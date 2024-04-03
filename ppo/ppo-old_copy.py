# import warnings
from typing import Any, Dict, Optional, Type, Union, Iterable
# import warnings
from warnings import warn
import tqdm

# import torch.nn
from gym import spaces
# import numpy as np
from numpy import mean as np_mean, zeros, float32, concatenate, ndarray
from numpy import max as np_max, min as np_min
from numpy import linalg
# import torch as th
from torch import Tensor, device, min, max, clamp, abs, exp, no_grad, mean, where, concatenate as th_concatenate, var, \
    empty, cuda
from torch.nn.utils import parameters_to_vector
from torch.linalg import vector_norm as th_norm
# from torch.nn import functional as F
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_

_tensor_or_tensors = Union[Tensor, Iterable[Tensor]]

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance_torch, get_schedule_fn


class PPO(OnPolicyAlgorithm):
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
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        ent_coef: Union[float, Schedule] = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        max_pol_loss: float = -0.0,
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

        super(PPO, self).__init__(
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
            batch_size=batch_size
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
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl
        # maybe?
        # benchmark = True
        self.last_params_store = None
        self.max_pol_loss = max_pol_loss

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(PPO, self)._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)
           
        self.ent_coef = get_schedule_fn(self.ent_coef)

    def _get_flat_gradient(self) -> float:
        # flat = []
        sequence_arrs = []
        # res = 0
        for p in self.policy.parameters():
            if p.grad is not None:
                # grad = p.grad.data.detach().cpu().numpy().ravel()
                sequence_arrs.append(p.grad.detach().ravel())
                # res += th_norm(p.grad.detach().ravel()).cpu().item()
            else:
                # grad = zeros(p.shape).ravel()
                # sequence_arrs.append(zeros(p.shape).ravel())
                pass

        # flat = parameters_to_vector(model.parameters()).grad.data.detach().cpu().numpy()
        # grads = parameters_to_vector(model.parameters()).grad
        # flat = where(grads is not None, grads, 0).detach().cpu().numpy()

        # flat = concatenate(sequence_arrs)
        # return linalg.norm(flat)
        return th_norm(th_concatenate(sequence_arrs)).item()
        # return res

    def _get_param_diff(self) -> float:
        flat = []
        if self.last_params_store is None:
            new_params = self.policy.parameters()
            for p in new_params:
                diff = zeros(p.shape, dtype=float32).ravel()
                flat = concatenate((flat, diff))
            self.last_params_store = parameters_to_vector(self.policy.parameters()).detach().clone()
            return linalg.norm(flat)
        else:
            new_params = parameters_to_vector(self.policy.parameters()).detach()
            # diff = (new_params - self.last_params_store).cpu().numpy()
            diff = new_params - self.last_params_store
            # flat = concatenate((flat, diff))
            self.last_params_store = new_params
            # return linalg.norm(diff)
            return th_norm(diff).item()

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
        ent_coef = self.ent_coef(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = None
        pg_losses, value_losses = None, None
        clip_fractions = None
        losses = None
        # gradients = empty(0)
        # param_diffs = empty(0)
        
        last_kl_div = None
        last_clip_fraction = None
        last_pg_loss = None
        # last_off_policy_penalty = 0

        continue_training = True

        progress_bar = tqdm.tqdm(desc=f"Training", total=self.n_epochs, leave=False,
                                 smoothing=0.01)

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = None
            # Do a complete pass on the rollout buffer
            progress_bar.set_description(f"Training, rollout buffer get | epoch")
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                progress_bar.set_description(f"Training, PPO calc | epoch")
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    # actions = rollout_data.actions.long().flatten()
                    actions = rollout_data.actions.int().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)
                
                progress_bar.set_description(f"Training, PPO eval acts | epoch")
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                progress_bar.set_description(f"Training, PPO norm advs | epoch")
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                progress_bar.set_description(f"Training, PPO calc ratio | epoch")
                log_prob_diff = log_prob - rollout_data.old_log_prob
                ratio = exp(log_prob_diff)

                # clipped surrogate loss
                # clamped_off_policy_ratio = clamp(ratio, 1 - clip_range, 1 + clip_range)
                # clamp_max = clamp(ratio, 1, None)
                # max_ratio_min = 1
                # max_ratio_max = max(ratio)
                # clamped_off_policy_ratio = ((1 + clip_range) - (1 - clip_range)) * ((ratio - ratio_min) / (ratio_max - ratio_min)) + (1 - clip_range)
                # clamped_off_policy_ratio_max = ((1 + clip_range) - 1) * ((clamp_max - ratio_min) / (ratio_max - ratio_min)) + 1
                # clamped_off_policy_ratio_max = max_ratio_min + ((clamp_max - max_ratio_min) / (max_ratio_max - max_ratio_min + 1e-5)) * clip_range
                
                # clamp_min = clamp(ratio, None, 1)
                # min_ratio_min = min(ratio)
                # min_ratio_max = 1
                # clamped_off_policy_ratio_min = (1 - (1 - clip_range)) * ((clamp_min - ratio_min) / (ratio_max - ratio_min)) + (1 - clip_range)
                # clamped_off_policy_ratio_min = min_ratio_max - (1 - ((clamp_min - min_ratio_min) / (min_ratio_max - min_ratio_min + 1e-5))) * clip_range
                
                # clamped_off_policy_ratio = clamped_off_policy_ratio_max * clamped_off_policy_ratio_min
                # part = 10/(clip_range/0.1)
                # off_policy_penalty = min(
                #     -part*clamped_off_policy_ratio + (1+part),
                #     part*clamped_off_policy_ratio + (1-part)
                # )
                
                progress_bar.set_description(f"Training, PPO policy loss | epoch")
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -min(policy_loss_1, policy_loss_2).mean()
                # policy_loss = -policy_loss_1.mean() * off_policy_penalty.mean()
                
                # simple kl spike prevention for now
                if policy_loss > self.max_pol_loss and epoch > 4:
                    del policy_loss
                    del policy_loss_1
                    del policy_loss_2
                    del ratio
                    del log_prob_diff
                    del advantages
                    del values
                    del log_prob
                    del entropy
                    del actions
                    break
                    
                if policy_loss.isnan().any():
                    del policy_loss
                    del policy_loss_1
                    del policy_loss_2
                    del ratio
                    del log_prob_diff
                    del advantages
                    del values
                    del log_prob
                    del entropy
                    del actions
                    self.policy.optimizer.zero_grad(set_to_none=True)
                    print(f"got policy loss NaN")
                    # break
                    raise ArithmeticError('Got nan in policy loss')

                # Logging
                # pg_loss_py = policy_loss.item()
                # pg_losses.append(pg_loss_py)
                if pg_losses is None:
                    pg_losses = policy_loss.detach().unsqueeze(0).float()
                else:
                    th_concatenate((pg_losses, policy_loss.detach().unsqueeze(0).float()))
                # last_pg_loss = pg_loss_py
                last_pg_loss = policy_loss
                # last_off_policy_penalty = mean(off_policy_penalty).float().item()
                # clip_fraction = mean((abs(ratio - 1) > clip_range).float()).item()
                progress_bar.set_description(f"Training, PPO clip frac | epoch")
                clip_fraction = mean((abs(ratio - 1) > clip_range).float())
                # clip_fractions.append(clip_fraction)
                if clip_fractions is None:
                    clip_fractions = clip_fraction.detach().unsqueeze(0)
                else:
                    th_concatenate((clip_fractions, clip_fraction.detach().unsqueeze(0)))
                last_clip_fraction = clip_fraction.detach()

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                progress_bar.set_description(f"Training, PPO val loss | epoch")
                value_loss = mse_loss(rollout_data.returns, values_pred)
                
                if value_loss.isnan().any():
                    del policy_loss
                    del policy_loss_1
                    del policy_loss_2
                    del value_loss
                    del ratio
                    del log_prob_diff
                    del advantages
                    del values
                    del log_prob
                    del entropy
                    del actions
                    self.policy.optimizer.zero_grad(set_to_none=True)
                    print(f"got value loss NaN")
                    # break
                    raise ArithmeticError('Got nan in value loss')
                    
                # value_losses.append(value_loss.item())
                if value_losses is None:
                    value_losses = value_loss.detach().unsqueeze(0)
                else:
                    th_concatenate((value_losses, value_loss.detach().unsqueeze(0)))

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -mean(-log_prob)
                else:
                    entropy_loss = -mean(entropy)

                # entropy_losses.append(entropy_loss.item())
                if entropy_losses is None:
                    entropy_losses = entropy_loss.detach().unsqueeze(0)
                else:
                    th_concatenate((entropy_losses, entropy_loss.detach().unsqueeze(0)))

                loss = policy_loss + ent_coef * entropy_loss + self.vf_coef * value_loss
                # losses.append(loss.item())
                if losses is None:
                    losses = loss.detach().unsqueeze(0).float()
                else:
                    th_concatenate((losses, loss.detach().unsqueeze(0).float()))

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with no_grad():
                    progress_bar.set_description(f"Training, PPO kl div | epoch")
                    # log_ratio = log_prob - rollout_data.old_log_prob
                    # approx_kl_div = mean((ratio - 1) - log_prob_diff).cpu().numpy()
                    approx_kl_div = mean((ratio - 1) - log_prob_diff)
                    # approx_kl_div_scalar = approx_kl_div.item()
                    # approx_kl_divs.append(approx_kl_div_scalar)
                    if approx_kl_divs is None:
                        approx_kl_divs = approx_kl_div.detach().unsqueeze(0)
                    else:
                        th_concatenate((approx_kl_divs, approx_kl_div.detach().unsqueeze(0)))
                    last_kl_div = approx_kl_div

                if self.target_kl is not None and approx_kl_div > self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                progress_bar.set_description(f"Training, backwards | epoch")
                self.policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # gradients.append(self._get_flat_gradient())

                # Clip grad norm
                # th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                if self.max_grad_norm != 0:
                    progress_bar.set_description(f"Training, clipping grads | epoch")
                    # self.policy: torch.nn.Module = self.policy.to("cpu", non_blocking=True)
                    # self.clip_grad_norm_test(self.policy.parameters(), self.max_grad_norm)
                    clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    # self.policy = self.policy.to(self.device, non_blocking=True)
                progress_bar.set_description(f"Training, optimizer step | epoch")
                self.policy.optimizer.step()
                
            # should be updated for each epoch
            self._n_updates += 1

                # param_diffs.append(self._get_param_diff())

            if not continue_training:
                break

            progress_bar.update(1)

        try:
            progress_bar.close()
        except:
            pass
            
        # if "cuda" in self.device:
        #     cuda.empty_cache()

        # self._n_updates += self.n_epochs
        explained_var = explained_variance_torch(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        if entropy_losses is not None:
            self.logger.record("train/entropy_loss", np_mean(entropy_losses.cpu().numpy()))
        if pg_losses is not None:
            self.logger.record("train/policy_gradient_loss", np_mean(pg_losses.cpu().numpy()))
        if value_losses is not None:
            self.logger.record("train/value_loss", np_mean(value_losses.cpu().numpy()))
        if approx_kl_divs is not None:
            self.logger.record("train/approx_kl", np_mean(approx_kl_divs.cpu().numpy()))
        if last_kl_div is not None:
            self.logger.record("train/last_kl_div", last_kl_div.item())  # new
        if clip_fractions is not None:
            self.logger.record("train/clip_fraction", np_mean(clip_fractions.cpu().numpy()))
        if last_clip_fraction is not None:
            self.logger.record("train/last_clip_fraction", last_clip_fraction.item())  # new
        if loss is not None:
            self.logger.record("train/loss", loss.item())
        # new
        if losses is not None:
            self.logger.record("train/loss_mean", np_mean(losses.cpu().numpy()))
        self.logger.record("train/returns_var", var(self.rollout_buffer.returns.flatten()).item())
        if last_pg_loss is not None:
            self.logger.record("train/last_policy_gradient_loss", last_pg_loss.item())
        # self.logger.record("train/last_off_policy_penalty", last_off_policy_penalty)

        # self.logger.record("train/gradients_mean", np_mean(gradients))
        # self.logger.record("train/gradients_max", np_max(gradients))
        # self.logger.record("train/gradients_min", np_min(gradients))

        # self.logger.record("train/params_diff_mean", np_mean(param_diffs))
        # self.logger.record("train/params_diff_max", np_max(param_diffs))
        # self.logger.record("train/params_diff_min", np_min(param_diffs))
        #
        self.logger.record("train/explained_variance", explained_var.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("config/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("config/clip_range_vf", clip_range_vf)

        # new
        self.logger.record("config/entropy_coef", ent_coef)
        self.logger.record("config/n_epochs", self.n_epochs)
        self.logger.record("config/batch_size", self.batch_size)
        self.logger.record("config/gamma", self.gamma)
        self.logger.record("config/gae_lambda", self.gae_lambda)
        self.logger.record("config/buffer_size", self.n_steps*self.n_envs)
        self.logger.record("config/max_grad_norm", self.max_grad_norm)
        #

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "PPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "PPO":

        return super(PPO, self).learn(
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
