import gym
env = gym.make('CarRacing-v0')
env.reset()

import time

for i in range(10000):
    env.render()
    env.step(env.action_space.sample())
    time.sleep(0.1)


