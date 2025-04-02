# import warnings
from typing import Any, Dict, Optional, Type, Union, Iterable
from warnings import warn
import tqdm
import gc
import copy
from functools import partial

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
import torch as th
from torch import nn
# from torch.nn import functional as F
from torch.autograd.functional import hessian
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.cuda import empty_cache, Stream, current_stream, CUDAGraph, graph, stream, synchronize
from math import isnan
from random import randint

# from stable_baselines3.common.buffers import PrioritizedExperienceReplay

# _tensor_or_tensors = Union[Tensor, Iterable[Tensor]]

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicyOptim, ACPolicyOptimSpecNorm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance_torch, get_schedule_fn, flat_grad, conjugate_gradient_solver
from stable_baselines3.common.distributions import kl_divergence
# from stable_baselines3.common.torch_utils import utils_cgn


class TRPO_Optim(OnPolicyAlgorithm):
    """
    Trust Region Policy Optimization (TRPO)

    Paper: https://arxiv.org/abs/1502.05477
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    and Stable Baselines (TRPO from https://github.com/hill-a/stable-baselines)

    Introduction to TRPO: https://spinningup.openai.com/en/latest/algorithms/trpo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate for the value function, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size for the value function
    :param gamma: Discount factor
    :param cg_max_steps: maximum number of steps in the Conjugate Gradient algorithm
        for computing the Hessian vector product
    :param cg_damping: damping in the Hessian vector product computation
    :param line_search_shrinking_factor: step-size reduction factor for the line-search
        (i.e., ``theta_new = theta + alpha^i * step``)
    :param line_search_max_iter: maximum number of iteration
        for the backtracking line-search
    :param n_critic_updates: number of critic updates per policy update
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param normalize_advantage: Whether to normalize or not the advantage
    :param target_kl: Target Kullback-Leibler divergence between updates.
        Should be small for stability. Values like 0.01, 0.05.
    :param sub_sampling_factor: Sub-sample the batch to make computation faster
        see p40-42 of John Schulman thesis http://joschu.net/docs/thesis.pdf
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`trpo_policies`
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[Union[ActorCriticPolicyOptim, ACPolicyOptimSpecNorm]]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 128,
        critic_batch_size: Optional[int] = None,
        minibatch_size: Optional[int] = None,
        n_epochs: int = 1,
        n_epochs_critic: int = 0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        max_clip: Optional[float] = None,
        clip_range_vf: Union[None, float, Schedule] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        # trpo
        cg_max_steps: int = 15,
        cg_damping: float = 0.1,
        line_search_shrinking_factor: float = 0.8,
        line_search_max_iter: int = 10,
        # sub_sampling_factor: int = 1,
        #
        max_grad_norm: float = 0.5,
        max_pol_loss: float = -0.0,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: float = 0.01,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(TRPO_Optim, self).__init__(
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
        else:
            self.critic_batch_size = self.batch_size
        if minibatch_size is not None:
            self.minibatch_size = minibatch_size
        else:
            self.minibatch_size = batch_size
        assert self.batch_size % self.minibatch_size == 0, "Minibatch size must divide batch size evenly for the CUDA graph implementation"

        self.max_clip = max_clip
        self.n_epochs = n_epochs
        self.n_epochs_critic = n_epochs_critic
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl
        self.last_params_store = None
        self.max_pol_loss = max_pol_loss
        # trpo
        self.cg_max_steps = cg_max_steps
        self.cg_damping = cg_damping
        
        self.line_search_shrinking_factor = line_search_shrinking_factor
        self.line_search_grow_factor = 2 - line_search_shrinking_factor
        self.line_search_max_iter = line_search_max_iter
        #

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(TRPO_Optim, self)._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def _compute_actor_grad(
        self, kl_div: th.Tensor, policy_objective: th.Tensor
    ) -> tuple[list[nn.Parameter], th.Tensor, th.Tensor, list[tuple[int, ...]]]:
        """
        Compute actor gradients for kl div and surrogate objectives.

        :param kl_div: The KL divergence objective
        :param policy_objective: The surrogate objective ("classic" policy gradient)
        :return: List of actor params, gradients and gradients shape.
        """
        # This is necessary because not all the parameters in the policy have gradients w.r.t. the KL divergence
        # The policy objective is also called surrogate objective
        policy_objective_gradients_list = []
        # Contains the gradients of the KL divergence
        grad_kl_list = []
        # Contains the shape of the gradients of the KL divergence w.r.t each parameter
        # This way the flattened gradient can be reshaped back into the original shapes and applied to
        # the parameters
        grad_shape: list[tuple[int, ...]] = []
        # Contains the parameters which have non-zeros KL divergence gradients
        # The list is used during the line-search to apply the step to each parameters
        actor_params: list[nn.Parameter] = []

        # BUG: this uses an insane amount of VRAM, must be a better way
        for param in self.policy.mlp_extractor.policy_net.parameters():
            # Skip parameters related to value function based on name
            # this work for built-in policies only (not custom ones)
            # if "value" in name:
            #     continue

            # For each parameter we compute the gradient of the KL divergence w.r.t to that parameter
            kl_param_grad, *_ = th.autograd.grad(
                kl_div,
                param,
                create_graph=True,
                # retain_graph=True,
                allow_unused=True,
                # only_inputs=True,
            )
            # If the gradient is not zero (not None), we store the parameter in the actor_params list
            # and add the gradient and its shape to grad_kl and grad_shape respectively
            if kl_param_grad is not None:
                # If the parameter impacts the KL divergence (i.e. the policy)
                # we compute the gradient of the policy objective w.r.t to the parameter
                # this avoids computing the gradient if it's not going to be used in the conjugate gradient step
                policy_objective_grad, *_ = th.autograd.grad(policy_objective, param, retain_graph=True)

                grad_shape.append(kl_param_grad.shape)
                grad_kl_list.append(kl_param_grad.reshape(-1))
                policy_objective_gradients_list.append(policy_objective_grad.reshape(-1).detach().clone())
                actor_params.append(param)

        # For each parameter we compute the gradient of the KL divergence w.r.t to that parameter
        # BUG: why does list(.parameters()) not work??? why does parameters_to_vector() create Nones for grads???
        # params = parameters_to_vector(self.policy.mlp_extractor.policy_net.parameters())
        # kl_param_grad, *_ = th.autograd.grad(
        #     kl_div,
        #     params,
        #     create_graph=True,
        #     # retain_graph=True,
        #     allow_unused=True,
        #     # only_inputs=True,
        # )
        # # If the gradient is not zero (not None), we store the parameter in the actor_params list
        # # and add the gradient and its shape to grad_kl and grad_shape respectively
        # if kl_param_grad is not None:
        #     # If the parameter impacts the KL divergence (i.e. the policy)
        #     # we compute the gradient of the policy objective w.r.t to the parameter
        #     # this avoids computing the gradient if it's not going to be used in the conjugate gradient step
        #     policy_objective_grad, *_ = th.autograd.grad(policy_objective, parameters_to_vector(self.policy.mlp_extractor.policy_net.parameters()), retain_graph=True)

        #     grad_shape.append(kl_param_grad.shape)
        #     grad_kl_list.append(kl_param_grad.reshape(-1))
        #     policy_objective_gradients_list.append(policy_objective_grad.reshape(-1))
        #     # actor_params.append(list(self.policy.mlp_extractor.policy_net.parameters()))
        #     actor_params = parameters_to_vector(self.policy.mlp_extractor.policy_net.parameters())

        # Gradients are concatenated before the conjugate gradient step
        policy_objective_gradients = th.cat(policy_objective_gradients_list)
        grad_kl = th.cat(grad_kl_list)
        return actor_params, policy_objective_gradients, grad_kl, grad_shape

    def hessian_vector_product(
        self, params: list[nn.Parameter], grad_kl: th.Tensor, vector: th.Tensor, retain_graph: bool = True
    ) -> th.Tensor:
        """
        Computes the matrix-vector product with the Fisher information matrix.

        :param params: list of parameters used to compute the Hessian
        :param grad_kl: flattened gradient of the KL divergence between the old and new policy
        :param vector: vector to compute the dot product the hessian-vector dot product with
        :param retain_graph: if True, the graph will be kept after computing the Hessian
        :return: Hessian-vector dot product (with damping)
        """
        # FIXME: I hate this
        jacobian_vector_product = (grad_kl * vector).sum()
        return flat_grad(jacobian_vector_product, params, retain_graph=retain_graph) + self.cg_damping * vector

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
        self._update_learning_rate([self.policy.value_optimizer, self.policy.policy_optimizer])
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        # entropy_losses = []
        value_losses = []
        approx_kl_divs = []
        
        last_kl_div = 0
        
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

            if not continue_training:
                break

            progress_bar.update(1)
            self._n_updates += 1
            # to replicate functionality of where it was originally
        
        postcompute_val_extract = parameters_to_vector(self.policy.mlp_extractor.value_net.parameters())
        postcompute_val_pol = parameters_to_vector(self.policy.value_net.parameters())
        postcompute_val = th_concatenate((postcompute_val_extract, postcompute_val_pol)).cpu()

        empty_cache()
        gc.collect()

        precompute_act_extract = parameters_to_vector(self.policy.mlp_extractor.policy_net.parameters())
        precompute_act_pol = parameters_to_vector(self.policy.action_net.parameters())
        precompute_act = th_concatenate((precompute_act_extract, precompute_act_pol)).cpu()

        # pg_losses_minibatch = []
        # clip_fractions_minibatch = []
        policy_objective_values = []
        line_search_results = []
        kl_divergences = []
        # approx_kl_divs_minibatch = []
        entropy_losses = []

        actor_params, policy_objective_gradient, kl_grad, grad_shape = None, None, None, None

        pol_grads = []
        kl_grads = []
        policy_objs_store = []
        kl_divs_store = []
        last_iter = 0

        # train for n_epochs epochs, both policy and critic
        if self.policy.policy_optimizer.param_groups[0]['lr'] != 0:
            np_seed = np_base_seed
            for epoch in range(self.n_epochs):
                # Do a complete pass on the rollout buffer
                progress_bar.set_description(f"Training, rollout buffer get | epoch")
                # batch count (minibatch count not included/counted)
                batch = 0
                np_seed += 1
                for rollout_data in self.rollout_buffer.get(self.batch_size, np_seed):
                # for rollout_data, final_minibatch in \
                #         self.rollout_buffer.get_minibatch(self.batch_size, self.minibatch_size, np_seed):
                    # Re-sample the noise matrix because the log_std has changed
                    if self.use_sde:
                        self.policy.reset_noise(self.batch_size)

                    progress_bar.set_description(f"Training, TRPO collect kl_div and pol_grad | epoch")

                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to long
                        actions = actions.int().flatten()

                    # policy_latent = self.policy.get_latent(rollout_data.observations)

                    with th.no_grad():
                        # old_distribution = self.policy._get_action_dist_from_latent(policy_latent).detach().clone()
                        old_distribution = copy.copy(self.policy.get_distribution(rollout_data.observations))

                    distribution = self.policy.get_distribution(rollout_data.observations)
                    # policy_latent = self.policy.get_latent(rollout_data.observations)
                    # distribution = self.policy._get_action_dist_from_latent(policy_latent)
                    # old_distribution = distribution.detach().clone()
                    # old_distribution = copy.copy(distribution)
                    log_prob = distribution.log_prob(actions)
                    entropy = distribution.entropy()
                    entropy_loss = -mean(entropy)

                    # log_prob, entropy = self.policy.get_entr_prob(rollout_data.observations, actions)
                    # Normalize advantage
                    advantages = rollout_data.advantages
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # ratio between old and new policy, should be one at the first iteration
                    log_prob_diff = log_prob - rollout_data.old_log_prob
                    ratio = exp(log_prob_diff)

                    policy_objective = (advantages * ratio).mean() + self.ent_coef * entropy_loss
                    entropy_losses.append(entropy_loss.item())

                    kl_div = kl_divergence(distribution, old_distribution).mean()

                    # policy_objs_store.append(policy_objective)
                    # kl_divs_store.append(kl_div)
                    
                    # compute actor_params and grad_shape once? and then average policy obj grads and kl grads over minibatches
                    #
                    # likely choice is if final minibatch then take the previous minibatches and compute one final time for the current one,
                    # then use those actor params and grad shape but mean all of the gradients from the minibatches
                    # if not final_minibatch:
                    #     _, policy_objective_gradient, kl_grad, _ = self._compute_actor_grad(kl_div, policy_objective)
                    #     pol_grads.append(policy_objective_gradient)
                    #     kl_grads.append(kl_grad)

                    # hessian_vector_product_fn = partial(self.hessian_vector_product, actor_params, grad_kl)

                    # search_direction = conjugate_gradient_solver(
                    #     hessian_vector_product_fn,
                    #     policy_objective_gradients,
                    #     max_iter=self.cg_max_steps
                    # )

                    # line_search_max_step_size = 2 * self.target_kl
                    # line_search_max_step_size /= th.matmul(
                    #     search_direction, hessian_vector_product_fn(search_direction, retain_graph=False)
                    # )
                    # line_search_max_step_size = th.sqrt(line_search_max_step_size)

                    # line_search_backtrack_coeff = 1.0
                    # original_actor_params = [param.detach().clone() for param in actor_params]

                    # is_line_search_success = False
                    # with th.no_grad():
                    #     for _ in range(self.line_search_max_iter):
                    #         start_idx = 0
                    #         for param, original_param, shape in zip(actor_params, original_actor_params, grad_shape):

                        
                    # # Entropy loss favor exploration
                    # if self.policy.policy_optimizer.param_groups[0]['lr'] != 0:
                    #     if entropy is None:
                    #         # Approximate entropy when no analytical form
                    #         entropy_loss = -mean(-log_prob)
                    #     else:
                    #         entropy_loss = -mean(entropy)

                    #     entropy_losses_minibatch.append(entropy_loss.item())
                    
                    # if isnan(last_kl_div):
                    #     raise ArithmeticError('Got nan in approx_kl_div')

                    # if self.target_kl is not None and last_kl_div > 1.5 * self.target_kl:
                    #     continue_training = False
                    #     if self.verbose >= 1:
                    #         print(f"Early stopping at epoch {epoch}, batch {batch} due to reaching max kl: {last_kl_div:.2f}")
                    #     break

                    # Optimization step
                    # progress_bar.set_description(f"Training, backwards | epoch")
                    # if self.policy.policy_optimizer.param_groups[0]['lr'] != 0:
                    #     final_policy_loss.backward()
                    
                    # this is where we calculate the actual batch (optimizer step and etc.)
                    # if final_minibatch:
                    actor_params, policy_objective_gradient, kl_grad, grad_shape = self._compute_actor_grad(kl_div, policy_objective)
                    # pol_grads.append(policy_objective_gradient)
                    # kl_grads.append(kl_grad)

                # grad_kl = th.stack(kl_grads).mean(0)
                grad_kl = kl_grad
                # policy_objective_gradients = th.stack(pol_grads).mean(0)
                policy_objective_gradients = policy_objective_gradient
                
                self.policy.policy_optimizer.zero_grad(set_to_none=True)

                progress_bar.set_description(f"Training, TRPO compute final grads | epoch")

                # final_kl_div = th.stack(kl_divs_store).mean(0)
                # final_policy_obj = th.stack(policy_objs_store).mean(0)
                # actor_params, policy_objective_gradients, grad_kl, grad_shape = self._compute_actor_grad(final_kl_div, final_policy_obj)

                hessian_vector_product_fn = partial(self.hessian_vector_product, actor_params, grad_kl)
                # hessian_vector_product_fn = partial(hessian, actor_params, grad_kl, vectorize=True)

                search_direction = conjugate_gradient_solver(
                    hessian_vector_product_fn,
                    policy_objective_gradients,
                    max_iter=self.cg_max_steps
                )

                line_search_max_step_size = 2 * self.target_kl
                line_search_max_step_size /= th.matmul(
                    search_direction, hessian_vector_product_fn(search_direction, retain_graph=False)
                )
                line_search_max_step_size = th.sqrt(line_search_max_step_size)

                line_search_backtrack_coeff = 1.0
                original_actor_params = [param.detach().clone() for param in actor_params]

                is_line_search_success = False
                bounced = False
                progress_bar.set_description(f"Training, TRPO line search | epoch")
                with th.no_grad():
                    for i in range(self.line_search_max_iter):
                        start_idx = 0
                        for param, original_param, shape in zip(actor_params, original_actor_params, grad_shape):
                            n_params = param.numel()
                            param.data = (
                                original_param.data
                                + line_search_backtrack_coeff
                                * line_search_max_step_size
                                * search_direction[start_idx : (start_idx + n_params)].view(shape)
                            )
                            start_idx += n_params

                        # new_policy_objs_store = []
                        new_policy_objective = None
                        # new_kl_divs_store = []
                        kl_div = None
                        # for rollout_data in self.rollout_buffer.get(self.batch_size, np_seed):
                        # # for rollout_data, final_minibatch in \
                        # #         self.rollout_buffer.get_minibatch(self.batch_size, self.minibatch_size, np_seed):
                        #     actions = rollout_data.actions
                        #     if isinstance(self.action_space, spaces.Discrete):
                        #         # Convert discrete action from float to long
                        #         actions = actions.int().flatten()

                        distribution = self.policy.get_distribution(rollout_data.observations)
                        log_prob = distribution.log_prob(actions)
                        entropy = distribution.entropy()
                        entropy_loss = -mean(entropy)

                        ratio = th.exp(log_prob - rollout_data.old_log_prob)

                        # advantages = rollout_data.advantages
                        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                        
                        new_policy_objective = (advantages * ratio).mean() + self.ent_coef * entropy_loss
                        # new_policy_objs_store.append(new_policy_objective)

                        kl_div = kl_divergence(distribution, old_distribution).mean()
                        # new_kl_divs_store.append(kl_div)

                        # new_policy_objective = th.stack(new_policy_objs_store).mean(0)
                        # kl_div = th.stack(new_kl_divs_store).mean(0)

                        if (kl_div < self.target_kl) and (new_policy_objective > policy_objective):
                            #dbg
                            approx_kl_div = mean((ratio - 1) - (log_prob - rollout_data.old_log_prob)).cpu().numpy()

                            if bounced and not is_line_search_success:
                                is_line_search_success = True
                                break
                            else:
                                bounced = True
                            is_line_search_success = True
                            # break
                            line_search_backtrack_coeff *= self.line_search_grow_factor
                        else:
                            is_line_search_success = False
                            line_search_backtrack_coeff *= self.line_search_shrinking_factor

                        last_iter = i

                    line_search_results.append(is_line_search_success)

                    if not is_line_search_success:
                        for param, original_param in zip(actor_params, original_actor_params):
                            param.data = original_param.data.clone()

                        policy_objective_values.append(policy_objective.item())
                        kl_divergences.append(0.0)
                    else:
                        policy_objective_values.append(new_policy_objective.item())
                        kl_divergences.append(kl_div.item())

                        # batch += 1
                        # # Clip grad norm
                        # if self.max_grad_norm != 0:
                        #     progress_bar.set_description(f"Training, clipping grads | epoch")
                        #     clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                        # progress_bar.set_description(f"Training, optimizer step | epoch")
                        # if self.policy.policy_optimizer.param_groups[0]['lr'] != 0:
                        #     self.policy.policy_optimizer.step()

                        # if self.policy.policy_optimizer.param_groups[0]['lr'] != 0:
                        #     self.policy.policy_optimizer.zero_grad(set_to_none=True)

                        # approx_kl_divs.append(np_mean(approx_kl_divs_minibatch))
                        # entropy_losses.append(np_mean(entropy_losses_minibatch))

                        # approx_kl_divs_minibatch = []
                        # entropy_losses_minibatch = []

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
        self.logger.record("train/policy_objective", np_mean(policy_objective_values))
        self.logger.record("train/kl_divergence_loss", np_mean(kl_divergences))
        self.logger.record("train/is_line_search_success", np_mean(line_search_results))
        self.logger.record("train/num_line_search_iters", last_iter)
        self.logger.record("train/value_loss", np_mean(value_losses))
        self.logger.record("train/first_value_loss", value_losses[0])
        self.logger.record("train/last_value_loss", value_losses[-1])
        # self.logger.record("train/approx_kl", np_mean(approx_kl_divs))
        # self.logger.record("train/last_kl_div", last_kl_div)  # new
        self.logger.record("train/entropy_loss", np_mean(entropy_losses))
        self.logger.record("train/critic_update_magnitude", val_dist)
        self.logger.record("train/actor_update_magnitude", act_dist)
        self.logger.record("train/returns_var", var(self.rollout_buffer.returns.flatten()).item())
        self.logger.record("train/explained_variance", explained_var.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        # self.logger.record("config/clip_range", clip_range)
        # if self.clip_range_vf is not None:
            # self.logger.record("config/clip_range_vf", clip_range_vf)

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

        return super(TRPO_Optim, self).learn(
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
