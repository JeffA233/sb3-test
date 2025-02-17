# import warnings
from typing import Any, Dict, Optional, Type, Union, Iterable
from warnings import warn
import tqdm
import gc

# import torch.nn
from gym import spaces
# import numpy as np
from numpy import mean as np_mean, zeros, float32, concatenate, ndarray
from numpy import max as np_max, min as np_min
from numpy import linalg
# import torch as th
from torch import Tensor, device, min, clamp, abs, exp, no_grad, mean, where, concatenate as th_concatenate, var, max, dist, topk, zeros as th_zeros, tensor, float32 as th_float32, sum as th_sum, isnan as th_isnan
from torch.nn.utils import parameters_to_vector
from torch.linalg import vector_norm as th_norm
# from torch.nn import functional as F
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.cuda import empty_cache, Stream, current_stream, CUDAGraph, graph, stream
from math import isnan
from random import randint

# from stable_baselines3.common.buffers import PrioritizedExperienceReplay

_tensor_or_tensors = Union[Tensor, Iterable[Tensor]]

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicyOptim
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance_torch, get_schedule_fn
# from stable_baselines3.common.torch_utils import utils_cgn


class PPO_Optim(OnPolicyAlgorithm):
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
    :param batch_size: batch size
    :param minibatch_size: minibatch size
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
        policy: Union[str, Type[ActorCriticPolicyOptim]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        n_steps_per: int = 2048,
        max_epoches_per: int = 100,
        batch_size: int = 64,
        critic_batch_size: Optional[int] = None,
        minibatch_size: Optional[int] = None,
        n_epochs: int = 10,
        n_epochs_critic: int = 0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        max_clip: Optional[float] = None,
        clip_range_vf: Union[None, float, Schedule] = None,
        ent_coef: float = 0.0,
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

        super(PPO_Optim, self).__init__(
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
        if critic_batch_size is not None:
            self.critic_batch_size = critic_batch_size
            self.critic_batch_size_cuda = tensor(self.critic_batch_size, device=self.device, dtype=th_float32)
        else:
            self.critic_batch_size = self.batch_size
            self.critic_batch_size_cuda = tensor(self.batch_size, device=self.device, dtype=th_float32)
        if minibatch_size is not None:
            self.minibatch_size = minibatch_size
            self.minibatch_size_cuda = tensor(self.minibatch_size, device=self.device, dtype=th_float32)
        else:
            self.minibatch_size = batch_size
            self.minibatch_size_cuda = tensor(self.batch_size, device=self.device, dtype=th_float32)
        assert self.batch_size % self.minibatch_size == 0, "Minibatch size must divide batch size evenly for the CUDA graph implementation"
        self.batch_div = (self.minibatch_size_cuda / self.critic_batch_size_cuda).to(dtype=th_float32, device=self.device)

        self.n_steps_per = n_steps_per
        self.max_epoches_per = max_epoches_per
        self.max_clip = max_clip
        self.n_epochs = n_epochs
        self.n_epochs_critic = n_epochs_critic
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl
        self.last_params_store = None
        self.max_pol_loss = max_pol_loss

        if _init_setup_model:
            self._setup_model()

        # CUDA graph tensor setup
        # value loss section
        # self.graph_obs = th_zeros(self.minibatch_size + self.rollout_buffer.obs_shape)
        self.value_loss_gr = None
        self.value_update_gr = None

        self.obs_gr_store = None
        self.returns_gr_store = None
        self.value_loss_cuda_store = None
        self.value_pred_gr_store = None

        self.attempted_gr = False

    def _setup_model(self) -> None:
        super(PPO_Optim, self)._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

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
        
    def value_loss_and_back(self):
        # values_pred = self.policy.predict_values(self.obs_gr_store)
        # latent_vf = self.policy.mlp_extractor.forward_critic(self.obs_gr_store)
        latent_vf = self.policy.mlp_extractor.value_net(self.obs_gr_store)
        # BUG: errors out at this part down below
        # it doesn't do _slow_forward() here for the module which seems like an error
        values_pred = self.policy.value_net(latent_vf)
        #
        # values_pred_flat = values_pred.flatten()
        values_pred_flat = values_pred.squeeze()
        value_loss = mse_loss(self.returns_gr_store, values_pred_flat)
        # value_loss = ((values_pred_flat - self.returns_gr_store) ** 2).mean()
        # final_value_loss = value_loss / (self.critic_batch_size_cuda / self.minibatch_size_cuda)
        final_value_loss = value_loss * self.batch_div
        self.value_loss_cuda_store = final_value_loss.detach().clone()
        final_value_loss.backward()

    def update_value_net(self):
        clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.value_optimizer.step()

    # run stream until final epoch and then capture on final epoch and save to vars
    def handle_graphs_val(self, progress_bar, value_losses):
        value_losses_minibatch = []
        np_base_seed = randint(0, 100)
        np_seed = np_base_seed

        # setup buffer to send to cpu for graph reasons
        self.rollout_buffer.device = "cpu"

        cuda_stream = Stream(device=self.device)
        cuda_stream.wait_stream(current_stream())
        # do warm-up 
        with stream(cuda_stream):
            for epoch in range(self.n_epochs_critic - 1):
                # Do a complete pass on the rollout buffer
                progress_bar.set_description(f"Training, rollout buffer get | epoch")
                # batch count (minibatch count not included/counted)
                np_seed += 1

                # do critic update calcs
                for rollout_data, final_minibatch in \
                        self.rollout_buffer.get_minibatch(self.batch_size, self.minibatch_size, np_seed):

                    progress_bar.set_description(f"Training, value loss | epoch")

                    if self.obs_gr_store is None:
                        self.obs_gr_store = rollout_data.observations.cuda()
                    else:
                        self.obs_gr_store.copy_(rollout_data.observations)

                    if self.returns_gr_store is None:
                        self.returns_gr_store = rollout_data.returns.cuda()
                    else:
                        self.returns_gr_store.copy_(rollout_data.returns)
                    self.value_loss_and_back()
                    value_losses_minibatch.append(self.value_loss_cuda_store.item())

                    # this is where we calculate the actual batch (optimizer step and etc.)
                    if final_minibatch:
                        # Clip grad norm
                        progress_bar.set_description(f"Training, optimizer step | epoch")
                        self.update_value_net()
                        self.policy.value_optimizer.zero_grad(set_to_none=True)

                        value_losses.append(np_mean(value_losses_minibatch))
                        value_losses_minibatch = []

                progress_bar.update(1)
                self._n_updates += 1
            
        current_stream().wait_stream(cuda_stream)

        # capture graphs
        for epoch in range(1):
            # Do a complete pass on the rollout buffer
            progress_bar.set_description(f"Training, rollout buffer get | epoch")
            # batch count (minibatch count not included/counted)
            np_seed += 1

            # do critic update calcs
            for rollout_data, final_minibatch in \
                    self.rollout_buffer.get_minibatch(self.batch_size, self.minibatch_size, np_seed):

                progress_bar.set_description(f"Training, value loss | epoch")

                self.obs_gr_store.copy_(rollout_data.observations)
                self.returns_gr_store.copy_(rollout_data.returns)
                if final_minibatch:
                    # BUG: this errors out for some reason
                    # cuda_stream.wait_stream(current_stream())
                    # with stream(cuda_stream):
                    self.value_loss_gr = CUDAGraph()
                    self.policy.value_optimizer.zero_grad(set_to_none=True)
                    with graph(self.value_loss_gr):
                        self.value_loss_and_back()
                    # current_stream().wait_stream(cuda_stream)
                    # self.value_loss_and_back()
                else:
                    self.value_loss_and_back()
                # self.value_loss_and_back()
                value_losses_minibatch.append(self.value_loss_cuda_store.item())

                # this is where we calculate the actual batch (optimizer step and etc.)
                if final_minibatch:
                    progress_bar.set_description(f"Training, optimizer step | epoch")
                    # cuda_stream.wait_stream(current_stream())
                    # with stream(cuda_stream):
                    self.value_update_gr = CUDAGraph()
                    with graph(self.value_update_gr):
                        self.update_value_net()
                        # self.update_value_net()
                    # current_stream().wait_stream(cuda_stream)
                    # self.update_value_net()

            progress_bar.update(1)
            self._n_updates += 1
        
        self.rollout_buffer.device = self.device
        self.attempted_gr = True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate([self.policy.value_optimizer, self.policy.policy_optimizer])
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        approx_kl_divs = []
        
        last_kl_div = 0
        last_clip_fraction = 0
        
        actual_epochs = 0

        continue_training = True

        progress_bar = tqdm.tqdm(desc=f"Training", total=self.n_epochs+self.n_epochs_critic, leave=False,
                                 smoothing=0.01)

        precompute_val_extract = parameters_to_vector(self.policy.mlp_extractor.value_net.parameters())
        precompute_val_pol = parameters_to_vector(self.policy.value_net.parameters())
        precompute_val = th_concatenate((precompute_val_extract, precompute_val_pol)).cpu()

        value_losses_minibatch = []
        np_base_seed = randint(0, 100)
        np_seed = np_base_seed

        # train critic only
        for epoch in range(self.n_epochs_critic):
            # Do a complete pass on the rollout buffer
            progress_bar.set_description(f"Training, rollout buffer get | epoch")
            # batch count (minibatch count not included/counted)
            batch = 0
            np_seed += 1

            # do critic update calcs
            for rollout_data, final_minibatch in \
                    self.rollout_buffer.get_minibatch(self.batch_size, self.minibatch_size, np_seed):
                progress_bar.set_description(f"Training, value loss | epoch")

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.critic_batch_size)

                values = self.policy.predict_values(rollout_data.observations)
                values = values.flatten()

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
                value_loss = mse_loss(rollout_data.returns, values_pred)
                # value_losses.append(value_loss.item())
                value_losses_minibatch.append(value_loss.item())

                if value_loss.isnan().any():
                    if self.policy.value_optimizer.param_groups[0]['lr'] != 0:
                        del values
                        self.policy.policy_optimizer.zero_grad(set_to_none=True)
                        self.policy.value_optimizer.zero_grad(set_to_none=True)
                        print(f"got value loss NaN")
                        # break
                        raise ArithmeticError('Got nan in value loss')

                final_value_loss = (self.vf_coef * value_loss) / (self.critic_batch_size / self.minibatch_size)

                # Optimization step
                progress_bar.set_description(f"Training, backwards | epoch")

                final_value_loss.backward()


                # this is where we calculate the actual batch (optimizer step and etc.)
                if final_minibatch:
                    batch += 1
                    if self.max_grad_norm != 0:
                        progress_bar.set_description(f"Training, clipping grads | epoch")
                        clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    progress_bar.set_description(f"Training, optimizer step | epoch")

                    self.policy.value_optimizer.step()
                    self.policy.value_optimizer.zero_grad(set_to_none=True)

                    value_losses.append(np_mean(value_losses_minibatch))
                    value_losses_minibatch = []

                # del rollout_data

            if not continue_training:
                break

            progress_bar.update(1)
            # actual_epochs += 1
            self._n_updates += 1
            # to replicate functionality of where it was originally
            # approx_kl_divs = []

        # # list to get CUDA Graphs working
        # # - removal of "if final_minibatch" or use two graphs with the split at the if statement (more likely easier)
        # # - switch to graph replay usage after the full train func has been used, populate CUDA graph on final epoch
        
        # if not self.attempted_gr:
        #     self.handle_graphs_val(progress_bar, value_losses)
        # else:
        #     # train critic only
        #     for epoch in range(self.n_epochs_critic):
        #         # Do a complete pass on the rollout buffer
        #         progress_bar.set_description(f"Training, rollout buffer get | epoch")
        #         # batch count (minibatch count not included/counted)
        #         np_seed += 1

        #         # do critic update calcs
        #         for rollout_data, final_minibatch in \
        #                 self.rollout_buffer.get_minibatch(self.batch_size, self.minibatch_size, np_seed):
        #             progress_bar.set_description(f"Training, value loss | epoch")

        #             self.obs_gr_store.copy_(rollout_data.observations)
        #             self.returns_gr_store.copy_(rollout_data.returns)
        #             if self.value_loss_gr:
        #                 self.value_loss_gr.replay()
        #             else:
        #                 self.value_loss_and_back()
        #             value_losses_minibatch.append(self.value_loss_cuda_store.item())

        #             # Optimization step
        #             # progress_bar.set_description(f"Training, backwards | epoch")

        #             # this is where we calculate the actual batch (optimizer step and etc.)
        #             if final_minibatch:
        #                 progress_bar.set_description(f"Training, optimizer step | epoch")
        #                 if self.value_update_gr:
        #                     self.value_update_gr.replay()
        #                 else:
        #                     self.update_value_net

        #                 # Clip grad norm
        #                 # progress_bar.set_description(f"Training, clipping grads | epoch")

        #                 value_losses.append(np_mean(value_losses_minibatch))
        #                 value_losses_minibatch = []

        #         progress_bar.update(1)
        #         self._n_updates += 1
        
        postcompute_val_extract = parameters_to_vector(self.policy.mlp_extractor.value_net.parameters())
        postcompute_val_pol = parameters_to_vector(self.policy.value_net.parameters())
        postcompute_val = th_concatenate((postcompute_val_extract, postcompute_val_pol)).cpu()

        empty_cache()
        gc.collect()

        precompute_act_extract = parameters_to_vector(self.policy.mlp_extractor.policy_net.parameters())
        precompute_act_pol = parameters_to_vector(self.policy.action_net.parameters())
        precompute_act = th_concatenate((precompute_act_extract, precompute_act_pol)).cpu()

        pg_losses_minibatch = []
        clip_fractions_minibatch = []
        approx_kl_divs_minibatch = []
        entropy_losses_minibatch = []

        # train for n_epochs epochs, both policy and critic
        if self.policy.policy_optimizer.param_groups[0]['lr'] != 0:
            np_seed = np_base_seed
            for epoch in range(self.n_epochs):
                # Do a complete pass on the rollout buffer
                progress_bar.set_description(f"Training, rollout buffer get | epoch")
                # batch count (minibatch count not included/counted)
                batch = 0
                np_seed += 1
                for rollout_data, final_minibatch in \
                        self.rollout_buffer.get_minibatch(self.batch_size, self.minibatch_size, np_seed):
                # for rollout_data in self.rollout_buffer.get(self.batch_size):
                    progress_bar.set_description(f"Training, PPO calc | epoch")
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to long
                        # actions = rollout_data.actions.long().flatten()
                        actions = actions.int().flatten()

                    # Re-sample the noise matrix because the log_std has changed
                    if self.use_sde:
                        self.policy.reset_noise(self.batch_size)

                    log_prob, entropy = self.policy.get_entr_prob(rollout_data.observations, actions)
                    # Normalize advantage
                    advantages = rollout_data.advantages
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # ratio between old and new policy, should be one at the first iteration
                    log_prob_diff = log_prob - rollout_data.old_log_prob
                    ratio = exp(log_prob_diff)

                    # DEBUG
                    # biggest_adv = topk(advantages, 10)
                    # smallest_adv = topk(advantages, 10, largest=False)
                    # biggest_ratio = topk(ratio, 10)
                    # smallest_ratio = topk(ratio, 10, largest=False)

                    # clipped surrogate loss
                    if self.policy.policy_optimizer.param_groups[0]['lr'] != 0:
                        policy_loss_1 = advantages * ratio
                        policy_loss_2 = advantages * clamp(ratio, 1 - clip_range, 1 + clip_range)
                        policy_loss = -min(policy_loss_1, policy_loss_2).mean()
                        
                        # New logging position
                        pg_losses_minibatch.append(policy_loss.item())

                        # simple kl spike prevention for now
                        if policy_loss > self.max_pol_loss and epoch > 0 and last_clip_fraction > 0.07:
                            # dbg
                            # pol_loss = -min(policy_loss_1, policy_loss_2)
                            # vals, indxs = max(pol_loss, 0)
                            # obs = rollout_data.observations[indxs].cpu().numpy()
                            #
                            del policy_loss
                            del policy_loss_1
                            del policy_loss_2
                            del ratio
                            del log_prob_diff
                            del advantages
                            del log_prob
                            del entropy
                            del actions
                            self.policy.policy_optimizer.zero_grad(set_to_none=True)
                            self.policy.value_optimizer.zero_grad(set_to_none=True)
                            print(f"got too large of a policy loss, skipping")
                            continue_training = False
                            break
                        
                        if policy_loss.isnan().any():
                            del policy_loss
                            del policy_loss_1
                            del policy_loss_2
                            del ratio
                            del log_prob_diff
                            del advantages
                            del log_prob
                            del entropy
                            del actions
                            self.policy.policy_optimizer.zero_grad(set_to_none=True)
                            self.policy.value_optimizer.zero_grad(set_to_none=True)
                            print(f"got policy loss NaN")
                            raise ArithmeticError('Got nan in policy loss')

                    clip_fraction = mean((abs(ratio - 1) > clip_range).float()).item()
                    
                    clip_fractions_minibatch.append(clip_fraction)
                    last_clip_fraction = clip_fraction
                    
                    # Calculate approximate form of reverse KL Divergence for early stopping
                    # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                    # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                    # and Schulman blog: http://joschu.net/blog/kl-approx.html
                    with no_grad():
                        approx_kl_div = mean((ratio - 1) - log_prob_diff).cpu().numpy()
                        approx_kl_div_scalar = approx_kl_div.item()
                        approx_kl_divs_minibatch.append(approx_kl_div_scalar)
                        last_kl_div = approx_kl_div_scalar
                    
                    # check if clip fraction was too high
                    if self.max_clip is not None and last_clip_fraction > self.max_clip:
                        print(f"hit clip_fraction > max_clip: {last_clip_fraction} on epoch: {actual_epochs} and batch: {batch}")
                        continue_training = False
                        break

                    # Entropy loss favor exploration
                    if self.policy.policy_optimizer.param_groups[0]['lr'] != 0:
                        if entropy is None:
                            # Approximate entropy when no analytical form
                            entropy_loss = -mean(-log_prob)
                        else:
                            entropy_loss = -mean(entropy)

                        # entropy_losses.append(entropy_loss.item())
                        entropy_losses_minibatch.append(entropy_loss.item())

                    if self.policy.policy_optimizer.param_groups[0]['lr'] != 0:
                        final_policy_loss = (policy_loss + self.ent_coef * entropy_loss) \
                                      / (self.batch_size / self.minibatch_size)
                    
                    if isnan(last_kl_div):
                        raise ArithmeticError('Got nan in approx_kl_div')

                    if self.target_kl is not None and last_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(f"Early stopping at epoch {epoch}, batch {batch} due to reaching max kl: {last_kl_div:.2f}")
                        break

                    # Optimization step
                    progress_bar.set_description(f"Training, backwards | epoch")
                    if self.policy.policy_optimizer.param_groups[0]['lr'] != 0:
                        final_policy_loss.backward()
                    
                    # this is where we calculate the actual batch (optimizer step and etc.)
                    if final_minibatch:
                        batch += 1
                        # Clip grad norm
                        if self.max_grad_norm != 0:
                            progress_bar.set_description(f"Training, clipping grads | epoch")
                            clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                        progress_bar.set_description(f"Training, optimizer step | epoch")
                        if self.policy.policy_optimizer.param_groups[0]['lr'] != 0:
                            self.policy.policy_optimizer.step()

                        if self.policy.policy_optimizer.param_groups[0]['lr'] != 0:
                            self.policy.policy_optimizer.zero_grad(set_to_none=True)

                        pg_losses.append(np_mean(pg_losses_minibatch))
                        clip_fractions.append(np_mean(clip_fractions_minibatch))
                        approx_kl_divs.append(np_mean(approx_kl_divs_minibatch))
                        entropy_losses.append(np_mean(entropy_losses_minibatch))

                        pg_losses_minibatch = []
                        clip_fractions_minibatch = []
                        approx_kl_divs_minibatch = []
                        entropy_losses_minibatch = []

                if not continue_training:
                    break

                progress_bar.update(1)
                self._n_updates += 1

        try:
            progress_bar.close()
        except:
            pass

        explained_var = explained_variance_torch(self.rollout_buffer.values.flatten(),
                                                 self.rollout_buffer.returns.flatten())
        
        postcompute_act_extract = parameters_to_vector(self.policy.mlp_extractor.policy_net.parameters())
        postcompute_act_pol = parameters_to_vector(self.policy.action_net.parameters())
        postcompute_act = th_concatenate((postcompute_act_extract, postcompute_act_pol)).cpu()

        val_dist = dist(precompute_val, postcompute_val).item()
        act_dist = dist(precompute_act, postcompute_act).item()

        # Logs
        if len(pg_losses) != 0:
            self.logger.record("train/entropy_loss", np_mean(entropy_losses))
        else:
            self.logger.record("train/entropy_loss", 0)
        if len(pg_losses) != 0:
            self.logger.record("train/policy_gradient_loss", np_mean(pg_losses))
            self.logger.record("train/first_policy_grad_loss", pg_losses[0])
            self.logger.record("train/last_policy_grad_loss", pg_losses[-1])
        else:
            self.logger.record("train/policy_gradient_loss", 0)
        self.logger.record("train/value_loss", np_mean(value_losses))
        self.logger.record("train/first_value_loss", value_losses[0])
        self.logger.record("train/last_value_loss", value_losses[-1])
        self.logger.record("train/approx_kl", np_mean(approx_kl_divs))
        self.logger.record("train/last_kl_div", last_kl_div)  # new
        self.logger.record("train/clip_fraction", np_mean(clip_fractions))
        self.logger.record("train/last_clip_fraction", last_clip_fraction)  # new
        self.logger.record("train/critic_update_magnitude", val_dist)
        self.logger.record("train/actor_update_magnitude", act_dist)
        self.logger.record("train/returns_var", var(self.rollout_buffer.returns.flatten()).item())
        self.logger.record("train/explained_variance", explained_var.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("config/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("config/clip_range_vf", clip_range_vf)

        # new
        self.logger.record("config/entropy_coef", self.ent_coef)
        self.logger.record("config/n_epochs", self.n_epochs)
        self.logger.record("config/batch_size", self.batch_size)
        self.logger.record("config/critic_batch_size", self.critic_batch_size)
        self.logger.record("config/minibatch_size", self.minibatch_size)
        self.logger.record("config/gamma", self.gamma)
        self.logger.record("config/gae_lambda", self.gae_lambda)
        self.logger.record("config/buffer_size", self.n_steps*self.n_envs)
        self.logger.record("config/max_grad_norm", self.max_grad_norm)
        self.logger.record("config/policy_lr", self.policy.policy_optimizer.param_groups[0]['lr'])
        self.logger.record("config/value_lr", self.policy.value_optimizer.param_groups[0]['lr'])
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
    ) -> "OnPolicyAlgorithm":

        return super(PPO_Optim, self).learn(
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
