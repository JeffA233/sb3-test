# This file is here just to define MlpPolicy/CnnPolicy
# that work for PPO
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    ActorCriticPolicyNorm,
    ActorCriticPolicyOptim,
    MultiInputActorCriticPolicy,
    register_policy,
)

MlpPolicy = ActorCriticPolicy
MlpNormPolicy = ActorCriticPolicyNorm
MlpOptimPolicy = ActorCriticPolicyOptim
CnnPolicy = ActorCriticCnnPolicy
MultiInputPolicy = MultiInputActorCriticPolicy

register_policy("MlpPolicy", ActorCriticPolicy)
register_policy("MlpNormPolicy", ActorCriticPolicyNorm)
register_policy("MlpOptimPolicy", ActorCriticPolicyOptim)
register_policy("CnnPolicy", ActorCriticCnnPolicy)
register_policy("MultiInputPolicy", MultiInputPolicy)
