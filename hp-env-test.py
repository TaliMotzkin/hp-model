import gymnasium as gym
import envs

seq = 'HPHHP'
env = gym.make('HPEnv-v0', seq=seq)

observations, _ = env.reset()
