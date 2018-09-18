from random import random, choice, randrange
import numpy as np
import gym
import time
import ai
import cv2

env = gym.make('Breakout-v0')



'''
env.reset()

for i in range(10000):
    env.render()
    env.step(env.action_space.sample())
    time.sleep(0.1)
'''



def process_image(image, resize=(80,80)):
    image = cv2.cvtColor(cv2.resize(image, resize), cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    return image

obs = env.reset()
print(process_image(obs).shape)


def choose_action(prediction, epsilon, action_space):
    actions = np.zeros([action_space])
    if random() <= epsilon:
        index = env.action_space.sample()
        actions[index] = 1
    else:
        pass

    return


def run():
    action_space = env.action_space.n
    ai.run_session()
    agent = ai.AI(action_space)
    observation = process_image(env.reset())


    for i in range(10000):

        done = False
        while not done:

            prediction = agent.predict()
            env.render()
            #action = choose_action(prediction, epsilon, action_space)
            #observation, reward, done, info = env.step(action)


