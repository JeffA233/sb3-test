# import time
from time import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import tqdm

# import gym
from gym.spaces import Discrete, Space, Box
from gym.spaces import Dict as Dict_Space
from numpy import ndarray
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBufferMC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean, safe_max, safe_min
from stable_baselines3.common.vec_env import VecEnv
# import torch as th
from torch import device, no_grad, Tensor, cuda, float32


class OnPolicyAlgorithmMC(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param policy_base: The base policy used by this method
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule],
            n_steps: int,
            gamma: float,
            gae_lambda: float,
            ent_coef: float,
            vf_coef: float,
            max_grad_norm: float,
            use_sde: bool,
            sde_sample_freq: int,
            policy_base: Type[BasePolicy] = ActorCriticPolicy,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            monitor_wrapper: bool = True,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[device, str] = "auto",
            _init_setup_model: bool = True,
            supported_action_spaces: Optional[Tuple[Space, ...]] = None,
            batch_size: int = 2048,
    ):

        super(OnPolicyAlgorithmMC, self).__init__(
            policy=policy,
            env=env,
            policy_base=policy_base,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            create_eval_env=create_eval_env,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None
        self.batch_size = batch_size

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, Dict_Space) else RolloutBufferMC

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        
        # print(self.policy_class)
        
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            device=self.device,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

# ---------------------------------- NEW WAY

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: RolloutBufferMC,
            n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        # experimentation with performance using different dtypes
        tensor_type = float32

        obs_tensor = obs_as_tensor(self._last_obs, self.device, dtype=tensor_type)

        # must set requires_grad to True for VERSION 2
        # obs_tensor.requires_grad = True
        # dist = self.policy.get_distribution(obs_tensor)
        # actions = dist.get_actions()
        # log_probs = dist.log_prob(actions)

        # CUDA graph policy only
        # VERSION 1
        # warmup
        latent_pi = self.policy.get_latent(obs_tensor)
        dist = self.policy._get_action_dist_from_latent(latent_pi)
        actions_tensor = dist.get_actions()
        log_probs_ten_cuda = dist.log_prob(actions_tensor)
        # log_probs_ten = log_probs_ten_cuda
        # actions_th, log_prob_th = self.policy.get_act_and_logprob(obs_tensor)
        # setting device here seems to be important?
        cuda_stream = cuda.Stream(device=self.device)
        cuda_stream.wait_stream(cuda.current_stream())
        with cuda.stream(cuda_stream):
            with no_grad():
                latent_pi = self.policy.get_latent(obs_tensor)
                dist = self.policy._get_action_dist_from_latent(latent_pi)
                actions_tensor = dist.get_actions()
                log_probs_ten_cuda = dist.log_prob(actions_tensor)
                # actions_th, log_prob_th = self.policy.get_act_and_logprob(obs_tensor)
        cuda.current_stream().wait_stream(cuda_stream)

        # capture
        cuda_gr = cuda.CUDAGraph()
        with cuda.graph(cuda_gr):
            with no_grad():
                latent_pi = self.policy.get_latent(obs_tensor)
                dist = self.policy._get_action_dist_from_latent(latent_pi)
                actions_tensor = dist.get_actions()
                log_probs_ten_cuda = dist.log_prob(actions_tensor)
                # actions_th, log_prob_th = self.policy.get_act_and_logprob(obs_tensor)

        # VERSION 2
        # policy_func = cuda.make_graphed_callables(self.policy.get_act_and_logprob, (obs_tensor,),
        #                                           allow_unused_input=True)
        # actions/log_prob test
        # dist = self.policy._get_action_dist_from_latent(latent_pi)
        # actions_th = dist.get_actions()
        # log_probs_th = dist.log_prob(actions_th)
        # cuda_stream_actprob = cuda.Stream(device=self.device)
        # cuda_stream_actprob.wait_stream(cuda.current_stream())
        # with cuda.stream(cuda_stream_actprob):
        #     with no_grad():
        #         actions_th = dist.get_actions()
        #         log_probs_th = dist.log_prob(actions_th)
        #
        # cuda_gr_actprob = cuda.CUDAGraph()
        # with cuda.graph(cuda_gr_actprob):
        #     with no_grad():
        #         actions_th = dist.get_actions()
        #         log_probs_th = dist.log_prob(actions_th)
        # ---------

        progress_bar = tqdm.tqdm(desc=f"Doing discarded steps, step count", total=800*self.n_envs,
                                 leave=False, smoothing=0.01)

        discard_steps = 800  # disable for now
        # discard some steps to keep on-policy
        while discard_steps < 800:
            if self.use_sde and self.sde_sample_freq > 0 and discard_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            cuda_gr.replay()
            # actions_th, log_prob_th = policy_func(obs_tensor)
            with no_grad():
                # dist = self.policy.get_distribution(obs_tensor)
                # dist = self.policy._get_action_dist_from_latent(latent_pi)
                # cuda_gr_actprob()
                # actions = actions_th.cpu().numpy()
                # actions = dist.get_actions().cpu().numpy()
                actions = actions_tensor.cpu().numpy()
                # actions: Tensor
            actions: ndarray

            # Rescale and perform action
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, Box):
                actions.clip(self.action_space.low, self.action_space.high, out=actions)

            new_obs, rewards, dones, infos = env.step(actions)

            obs_tensor.copy_(obs_as_tensor(new_obs, self.device, dtype=tensor_type), non_blocking=True)

            # Give access to local variables
            # callback.update_locals(locals())
            # if callback.on_step() is False:
            #     return False

            discard_steps += 1

            self._last_obs, self._last_episode_starts = new_obs, dones

            progress_bar.update(self.n_envs)

        progress_bar.close()
        # collection stuff -------------------------------------
        progress_bar = tqdm.tqdm(desc=f"Collecting rollouts, step count", total=n_rollout_steps * self.n_envs,
                                 leave=False, smoothing=0.01)

        # now we collect rollouts
        # while n_steps < n_rollout_steps:
        # uncomment this to use only full episodes when buffer is configured
        while not self.rollout_buffer.episodes_done:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            # get obs_tensor from thread
            # t2.join()
            # obs_tensor = que.get()

            cuda_gr.replay()
            # actions_th, log_prob_th = policy_func(obs_tensor)
            with no_grad():
                # dist = self.policy.get_distribution(obs_tensor)
                # dist = self.policy._get_action_dist_from_latent(latent_pi)
                # cuda_gr_actprob()
                # actions = actions_th.cpu().numpy()
                # log_probs = log_probs_th.cpu().numpy()
                # actions = dist.get_actions()
                # log_probs = dist.log_prob(actions)
                # log_probs = log_probs.cpu()
                log_probs = log_probs_ten_cuda.cpu()
                # actions: Tensor
            # actions: Tensor
            # actions = actions.cpu().numpy()
            actions = actions_tensor.cpu().numpy()
            actions: ndarray

            # Rescale and perform action
            # clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, Box):
                # clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
                actions.clip(self.action_space.low, self.action_space.high, out=actions)

            new_obs, rewards, dones, infos = env.step(actions)

            # t2 = threading.Thread(target=obs_func, args=(que, new_obs))
            # t2.start()
            obs_tensor.copy_(obs_as_tensor(new_obs, self.device, dtype=tensor_type), non_blocking=True)

            self.num_timesteps += env.num_envs
            # if isinstance(env.num_envs, ndarray) or isinstance(self.num_timesteps, ndarray):
            #     print("found array in timestep count, change to not get overflow")

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            # for idx, done in enumerate(dones):
            #     if (
            #             done
            #             and infos[idx].get("terminal_observation") is not None
            #             and infos[idx].get("TimeLimit.truncated", False)
            #     ):
            #         terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
            #         with no_grad():
            #             terminal_value = self.policy.predict_values(terminal_obs)[0]
            #         rewards[idx] += self.gamma * terminal_value
            #         # we don't want this to happen right now
            #         print("bootstrapped timeout with value net")
                    
            rollout_buffer.add_no_val(self._last_obs, actions, rewards, self._last_episode_starts, log_probs)
            # rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, None, None)
            self._last_obs, self._last_episode_starts = new_obs, dones
            # self._last_episode_starts = dones

            progress_bar.update(self.n_envs)

        progress_bar.set_description(f"Inferring values")

        # rollout_buffer.set_final_obs(self._last_obs)

        # compute values of the rollout and get the final value for calculation,
        # faster than computing a value for each obs while collecting rollouts as what original did
        # with no_grad():
        #     for rollout_data in rollout_buffer.get_non_rand(self.minibatch_size // self.n_envs):

        #         # Re-sample the noise matrix because the log_std has changed
        #         if self.use_sde:
        #             self.policy.reset_noise(self.batch_size)

        #         values = self.policy.predict_values(rollout_data.observations)
        #         rollout_buffer.set_values(values)
        #     # Compute value for the last timestep
        #     # t2.join()
        #     # obs_tensor = que.get()
        #     value = self.policy.predict_values(obs_tensor)

        progress_bar.set_description(f"Calculating returns")
        rollout_buffer.compute_returns_and_advantage(dones=dones.astype(int).astype(float))

        callback.update_locals(locals())

        callback.on_rollout_end()
        # t2.join()
        progress_bar.close()

        return True

# ---------------------------------- NEW WAY

# ---------------------------------- OLD WAY

    # def collect_rollouts(
    #     self,
    #     env: VecEnv,
    #     callback: BaseCallback,
    #     rollout_buffer: RolloutBuffer,
    #     n_rollout_steps: int,
    # ) -> bool:
    #     """
    #     Collect experiences using the current policy and fill a ``RolloutBuffer``.
    #     The term rollout here refers to the model-free notion and should not
    #     be used with the concept of rollout used in model-based RL or planning.
    #
    #     :param env: The training environment
    #     :param callback: Callback that will be called at each step
    #         (and at the beginning and end of the rollout)
    #     :param rollout_buffer: Buffer to fill with rollouts
    #     :param n_rollout_steps: Number of experiences to collect per environment
    #     :return: True if function returned with at least `n_rollout_steps`
    #         collected, False if callback terminated rollout prematurely.
    #     """
    #     assert self._last_obs is not None, "No previous observation was provided"
    #     # Switch to eval mode (this affects batch norm / dropout)
    #     self.policy.set_training_mode(False)
    #
    #     n_steps = 0
    #     rollout_buffer.reset()
    #     # Sample new weights for the state dependent exploration
    #     if self.use_sde:
    #         self.policy.reset_noise(env.num_envs)
    #
    #     callback.on_rollout_start()
    #
    #     progress_bar = tqdm.tqdm(desc=f"Collecting rollouts, step count", total=n_rollout_steps * self.n_envs,
    #                              leave=False, smoothing=0.01)
    #
    #     while n_steps < n_rollout_steps:
    #         if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
    #             # Sample a new noise matrix
    #             self.policy.reset_noise(env.num_envs)
    #
    #         with no_grad():
    #             # Convert to pytorch tensor or to TensorDict
    #             obs_tensor = obs_as_tensor(self._last_obs, self.device)
    #             actions, values, log_probs = self.policy.forward(obs_tensor)
    #         actions = actions.cpu().numpy()
    #
    #         # Rescale and perform action
    #         clipped_actions = actions
    #
    #         if isinstance(self.action_space, Box):
    #             if self.policy.squash_output:
    #                 # Unscale the actions to match env bounds
    #                 # if they were previously squashed (scaled in [-1, 1])
    #                 clipped_actions = self.policy.unscale_action(clipped_actions)
    #             else:
    #                 # Otherwise, clip the actions to avoid out of bound error
    #                 # as we are sampling from an unbounded Gaussian distribution
    #                 clipped_actions = clip(actions, self.action_space.low, self.action_space.high)
    #
    #         new_obs, rewards, dones, infos = env.step(clipped_actions)
    #
    #         self.num_timesteps += env.num_envs
    #
    #         # Give access to local variables
    #         callback.update_locals(locals())
    #         if callback.on_step() is False:
    #             return False
    #
    #         self._update_info_buffer(infos)
    #         n_steps += 1
    #
    #         if isinstance(self.action_space, Discrete):
    #             # Reshape in case of discrete action
    #             actions = actions.reshape(-1, 1)
    #
    #         # Handle timeout by bootstraping with value function
    #         # see GitHub issue #633
    #         for idx, done in enumerate(dones):
    #             if (
    #                 done
    #                 and infos[idx].get("terminal_observation") is not None
    #                 and infos[idx].get("TimeLimit.truncated", False)
    #             ):
    #                 terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
    #                 with no_grad():
    #                     terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
    #                 rewards[idx] += self.gamma * terminal_value
    #
    #         rollout_buffer.add(
    #             self._last_obs,  # type: ignore[arg-type]
    #             actions,
    #             rewards,
    #             self._last_episode_starts,  # type: ignore[arg-type]
    #             values,
    #             log_probs,
    #         )
    #         self._last_obs = new_obs  # type: ignore[assignment]
    #         self._last_episode_starts = dones
    #
    #         progress_bar.update(self.n_envs)
    #
    #     with no_grad():
    #         # Compute value for the last timestep
    #         values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]
    #
    #     progress_bar.set_description(f"Calculating returns")
    #     rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
    #
    #     callback.update_locals(locals())
    #
    #     callback.on_rollout_end()
    #     progress_bar.close()
    #
    #     return True

# ---------------------------------- OLD WAY

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "OnPolicyAlgorithm",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithmMC":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
            tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time() - self.start_time))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    ep_rew_mean = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
                    ep_len_mean = safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer])
                    # new
                    ep_rew_max = safe_max([ep_info["r"] for ep_info in self.ep_info_buffer])
                    ep_rew_min = safe_min([ep_info["r"] for ep_info in self.ep_info_buffer])
                    # ---
                    self.logger.record("rollout/ep_rew_mean", ep_rew_mean)
                    # new
                    self.logger.record("rollout/ep_rew_max", ep_rew_max)
                    self.logger.record("rollout/ep_rew_min", ep_rew_min)
                    # ---
                    self.logger.record("rollout/ep_len_mean", ep_len_mean)
                    # new
                    self.logger.record("rollout/ep_rew_per_len", ep_rew_mean / ep_len_mean)
                    #
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")

            self.train()

            if log_interval is not None and iteration % log_interval == 0:
                self.logger.dump(step=self.num_timesteps)

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
