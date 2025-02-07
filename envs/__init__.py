from gymnasium.envs.registration import register

register(
    id="HPEnv-v0",  # Unique ID
    entry_point="envs.hp-env:HPEnv",  # Path to class
)
