import gymnasium as gym

from envs import Action

seq = 'PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH'
env = gym.make('HPEnv_v0', seq=seq)

observations, _ = env.reset()

print(env.step(Action.LEFT.value))
print(env.step(Action.LEFT.value))
print(env.step(Action.RIGHT.value))
print(env.step(Action.RIGHT.value))
print(env.step(Action.LEFT.value))
print(env.step(Action.LEFT.value))
print(env.step(Action.RIGHT.value))
print(env.step(Action.RIGHT.value))
print(env.step(Action.LEFT.value))
print(env.step(Action.FORWARD.value))
print(env.step(Action.FORWARD.value))
print(env.step(Action.RIGHT.value))
print(env.step(Action.FORWARD.value))
print(env.step(Action.RIGHT.value))
print(env.step(Action.FORWARD.value))
print(env.step(Action.RIGHT.value))
print(env.step(Action.LEFT.value))
print(env.step(Action.LEFT.value))
print(env.step(Action.RIGHT.value))
print(env.step(Action.RIGHT.value))
print(env.step(Action.LEFT.value))
print(env.step(Action.LEFT.value))
print(env.step(Action.RIGHT.value))
print(env.step(Action.RIGHT.value))
print(env.step(Action.LEFT.value))
print(env.step(Action.LEFT.value))
print(env.step(Action.RIGHT.value))
print(env.step(Action.LEFT.value))
print(env.step(Action.FORWARD.value))
print(env.step(Action.LEFT.value))
print(env.step(Action.FORWARD.value))
print(env.step(Action.FORWARD.value))
print(env.step(Action.RIGHT.value))
print(env.step(Action.LEFT.value))
print(env.step(Action.LEFT.value))
print(env.step(Action.RIGHT.value))
print(env.step(Action.RIGHT.value))
print(env.step(Action.LEFT.value))
print(env.step(Action.LEFT.value))
print(env.step(Action.RIGHT.value))
print(env.step(Action.LEFT.value))
print(env.step(Action.LEFT.value))
print(env.step(Action.FORWARD.value))
print(env.step(Action.FORWARD.value))
print(env.step(Action.FORWARD.value))
print(env.step(Action.FORWARD.value))
