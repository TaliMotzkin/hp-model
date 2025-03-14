from gymnasium.envs.registration import register
from .hp_env import *

register(
    id="HPEnv_v0",  # Unique ID
    entry_point="envs.hp_env:HPEnv",  # Path to class
)

register(
    id="HPEnvGeneral",  # Unique ID
    entry_point="envs.hp_env_general:HPEnvGeneral",  # Path to class
)