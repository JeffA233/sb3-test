# Summary
Test repo for various personal experiments mainly for PPO. Effectively a Frankensteined stable-baselines3 of various updates with a substantial number of performance uplift changes. Also has some random attempts at modifying PPO with various ideas.

## Added Features
- PyTorch's CUDA graph tool is implemented for [OnPolicyAlgorithm](https://github.com/JeffA233/sb3-test/blob/master/common/on_policy_algorithm.py#L187) to speed up collecting rollouts. Currently PyTorch does not appear to support distributions with CUDA graphs so only the FFN is captured.
  - In the future, the critic can likely be captured too which will speed up calculating the values.

- Experimental rollout buffer [functionality](https://github.com/JeffA233/sb3-test/blob/master/common/buffers.py#L534) (also see the rest of the buffer) has been added that collects steps until either every vectorized environment has sent done or until the buffer has run out of room. In theory, this will help compute advantages more accurately especially if the final step is not otherwise done across all environments (which is unlikely).
  - This can be easily disabled via setting [max_mult](https://github.com/JeffA233/sb3-test/blob/master/common/buffers.py#L386) to one.
  - Additionally, the size of the buffer can be increased by the multiplier to allow for more or less headroom for the advantage calculation. Must be an integer for now.
  - Likely only more useful when the vectorized envs are very computationally inexpensive. 

- Batching values after collecting rollouts has also been [implemented](https://github.com/JeffA233/sb3-test/blob/master/common/on_policy_algorithm.py#L361) since computing values at each step is slower than batching especially when using a GPU. It is admittedly a bit awkward as it has to reformat the state's shape to pass to the critic as an input and then reformat the values again to put into the buffer.

- [ppo_split_optim](ppo_split_optim) has a separate optimizer for the critic and the policy.
  - This also allowed for the critic and the policy to be updated separately, increasing the memory usage headroom especially for VRAM-limited cases with larger batch sizes or network sizes.
  - Separate learning rates can now be applied as well as separate optimizers and optimizer configurations.
  - Note that shared networks will not work as intended when using this as they were not meant to be supported.

- Various other experiments are also in here but they should be deemed WIP and/or functionally not useful. 