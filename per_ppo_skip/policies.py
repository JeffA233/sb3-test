# This file is here just to define MlpPolicy/CnnPolicy
# that work for PER_PPO
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    ActorCriticPolicySkip,
    MultiInputActorCriticPolicy,
    register_policy,
)

MlpPolicySkip = ActorCriticPolicySkip
MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy
MultiInputPolicy = MultiInputActorCriticPolicy

register_policy("MlpPolicy", ActorCriticPolicy)
register_policy("MlpPolicySkip", ActorCriticPolicySkip)
register_policy("CnnPolicy", ActorCriticCnnPolicy)
register_policy("MultiInputPolicy", MultiInputPolicy)
