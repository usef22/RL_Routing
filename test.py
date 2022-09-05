import time
import numpy as np
import gym
import gym_novel_gridworlds

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from config import *

env = gym.make('NovelGridworld-v0')

# Load the trained agent

model = PPO2.load(model_name)

random = True

if random == True:
    for i_episode in range(100):
        print("EPISODE STARTS")
        obs = env.reset()
        # obs = env.reset(table=(5, 5), Agent=(8, 8))

        for i in range(1000):
            time.sleep(0.3)
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            env.render()
            if i_episode == 0 and i == 0:
                time.sleep(10)
            print("Episode #: " + str(i_episode) + ", step: " + str(i) + ", reward: ", reward)
            # End the episode if agent is dead
            if done:
                print("Episode #: "+str(i_episode)+" finished after "+str(i)+" timesteps\n")
                break

else:
    obs = env.reset(table=(5, 5), Agent=(8, 8))
    for i in range(1000):
        time.sleep(0.3)
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()

        print(", step: " + str(i) + ", reward: ", reward)
        # End the episode if agent is dead
        if done:
            print(" finished after " + str(i) + " timesteps\n")
            break