from stable_baselines3 import PPO
import pathlib
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
import glob
import os
import time


class Agent:
    def __init__(self):
        custom_objects = {
            "lr_schedule": 0.000001,
            "clip_range": .02,
            "n_envs": 1,
            "device": "cpu"
        }
        list_of_files = glob.glob('./models/*.zip')
        latest_file = max(list_of_files, key=os.path.getctime)
        latest_file = os.path.splitext(latest_file)[0]
        print(f"loading model {latest_file}")
        self.actor = PPO.load(f"{latest_file}", custom_objects=custom_objects)
        self.parser = KBMAction()

    def act(self, state):
        action = self.actor.predict(state, deterministic=True)
        x = self.parser.parse_actions(action[0], state)

        return x[0]
