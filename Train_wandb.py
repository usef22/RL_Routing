import os
import time
import wandb
import pandas as pd
import matplotlib.pyplot as plt

import gym
import gym_novel_gridworlds
from gym_novel_gridworlds.wrappers import SaveTrajectories, LimitActions
from gym_novel_gridworlds.observation_wrappers import LidarInFront, AgentMap
from gym_novel_gridworlds.novelty_wrappers import inject_novelty

import numpy as np

from stable_baselines.common.env_checker import check_env

from stable_baselines import PPO2, A2C, DQN
from stable_baselines.gail import ExpertDataset

from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import make_vec_env
from stable_baselines.gail import ExpertDataset


from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.results_plotter import load_results, ts2xy
from config import *

class RenderOnEachStep(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    """

    def __init__(self, env):
        super(RenderOnEachStep, self).__init__()
        self.env = env

    def _on_step(self):
        self.env.render()
        # time.sleep(0.5)

def ts2epi_len(timesteps):
    """
    Decompose a timesteps variable to x ans ys
    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """
    l_var = timesteps.l.values
    return l_var

def ts2epi_time(timesteps):
    "total time after end of episode"

    t_var = timesteps.t.values
    return t_var

for run in range(10):

    wandb.init(project="time")
    config = wandb.config

    # config.TEST_TYPE = "STEP"
    config.TEST_TYPE = "CUMULATIVE"
    config.Baseline = "A2C"
    # config.Map_Set = "Traffic Test"
    # config.STEP_REWARD = 30
    # config.CUMULATIVE_REWARD = 50
    # config.COMPLETION_REWARD = 100

    config.HighBW = 5
    config.BW_Req = 1.2
    config.BW_Maps = 3
    config.Punish = 600

    config.timesteps = 300000


    class SaveOnBestTrainingRewardCallback(BaseCallback):
        """
        Callback for saving a model (the check is done every ``check_freq`` steps)
        based on the training reward (in practice, we recommend using ``EvalCallback``).

        """

        def __init__(self, check_freq, log_dir, model_name):
            super(SaveOnBestTrainingRewardCallback, self).__init__()

            self.check_freq = check_freq
            self.log_dir = log_dir
            self.save_path = os.path.join(log_dir, model_name)
            self.best_mean_reward = -np.inf

        def _on_step(self):
            if self.n_calls % self.check_freq == 0:
                # Retrieve training reward
                timestamps = load_results(self.log_dir)
                traffic_file = "Logs/time_sheet.txt"
                f = open(traffic_file, "r")
                lines = f.readlines()
                lines = lines[2:]

                lines_ = []
                for line in lines:
                    line = line.replace("\n", "")
                    lines_.append(float(line))

                x, y = ts2xy(timestamps, 'timesteps')
                if len(x) > 0:
                    ##print(len(timestamps))
                    ##print("________________________________________________________________________")
                    ##print(timestamps)
                    # Mean training reward over the last 100 episodes
                    mean_reward = np.mean(y[-100:])
                    med_reward = np.median(y[-10:])
                    ep_reward = y[-1]
                    #wandb.log({"mean reward": mean_reward})
                    #wandb.log({"reward (median)": med_reward})
                    #wandb.log({"reward": ep_reward})
                    ## print(x, y)
                    episode_lengths = ts2epi_len(timestamps)
                    episode_length = episode_lengths
                    mean_length = np.mean(episode_length[-100:])
                    med_length = np.median(episode_length[-10:])
                    ep_length = episode_length[-1]
                    wandb.log({"mean ep length": mean_length})
                    wandb.log({"median ep length": med_length})
                    wandb.log({"ep length": ep_length})

                    ##if len(episode_length) < 100:
                    ##    ep_length = np.mean(episode_length)
                    ##else:
                    ##    ep_length = "{:.2f}".format(np.mean(episode_length[-100]))
                    #episode_time = ts2epi_time(timestamps)
                    #total_time = episode_time[-1]

                    lines_
                    ep_time = lines_[-1]
                    mean_time = np.mean(lines_[-100:])
                    med_time = np.median(lines_[-10:])
                    wandb.log({"ep time": ep_time})
                    wandb.log({"mean ep time": mean_time})
                    wandb.log({"median ep time": med_time})

                    print(ep_time)
                    print(lines_[-1])
                    print(mean_time)
                    print(med_time)



                    ##print(episode_length)
                    ##print(y)
                    ##print(mean_reward)
                    ##print(ep_length)
                    # New best model, you could save the agent here
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        print("Saving new best model to {}".format(self.save_path))
                        self.model.save(self.save_path)


    class RemapActionOnStep(BaseCallback):
        """
        Callback for saving a model (the check is done every ``check_freq`` steps)
        based on the training reward (in practice, we recommend using ``EvalCallback``).

        """

        def __init__(self, env, step_num):
            super(RemapActionOnStep, self).__init__()
            self.env = env
            self.step_num = step_num

        def _on_step(self):
            if self.n_calls == self.step_num:
                # self.env = remap_action(self.env)
                self.env.remap_action()
                # self.env = inject_novelty(self.env, 'firewall', 'hard', '', '')


    if __name__ == "__main__":
        env_id = 'NovelGridworld-v0'  # 'NovelGridworld-v1'
        # timesteps = 600000  # 200000
        experiment_dir = 'results'
        experiment_code1 = env_id + '_' + str(timesteps)
        reward_type = None
        if TEST_TYPE == "STEP":
            reward_type = STEP_REWARD
        elif TEST_TYPE == "CUMULATIVE":
            reward_type = CUMULATIVE_REWARD
        experiment_code2 =  "_" + str(TEST_TYPE) + "_" + str(reward_type) + "_" + str(COMPLETION_REWARD)
        model_code = experiment_code1 + experiment_code2
        log_dir = experiment_dir + os.sep + env_id
        pretrain = False

        os.makedirs(log_dir, exist_ok=True)

        env = gym.make(env_id)
        # env = LimitActions(env, {'Forward', 'Left', 'Right', 'Break', 'Craft_bow'})
        # env = LidarInFront(env)
        # env = inject_novelty(env, 'breakincrease', 'hard', '', '')

        env = Monitor(env, log_dir)
        # callback = RenderOnEachStep(env)
        callback = SaveOnBestTrainingRewardCallback(1000, log_dir, model_code + '_best_model')
        # callback = RemapActionOnEachStep(env, 50000)

        # multiprocess environment
        # env = make_vec_env('NovelGridworld-v0', n_envs=4)
        check_env(env, warn=True)

        # Optional: PPO2 requires a vectorized environment to run
        # the env is now wrapped automatically when passing it to the constructor
        # env = DummyVecEnv([lambda: env])

        model = A2C("MlpPolicy", env, verbose=1)

        # env = DummyVecEnv([lambda: env])
        # model = PPO2.load('NovelGridworld-Bow-v0_200000_8beams0filled11hypotenuserange3items_in_360degrees_best_model', env)

        # Pretrain the model from human recored dataset
        # specify `traj_limitation=-1` for using the whole dataset
        if pretrain:
            dataset = ExpertDataset(expert_path='expert_NovelGridworld-v0_10demos.npz', traj_limitation=-1, batch_size=128)
            model.pretrain(dataset, n_epochs=2000)
            model.save(log_dir + os.sep + model_code)

        # model.learn(total_timesteps=timesteps)
        model.learn(total_timesteps=timesteps, callback=callback)

        model.save(log_dir + os.sep + model_code + '_last_model')


    wandb.finish()