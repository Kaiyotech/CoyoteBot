import rlgym
from mybots_utils.mybots_rewards import *
from mybots_utils.mybots_statesets import *
from mybots_utils.mybots_terminals import *
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from rlgym_tools.extra_obs.advanced_stacker import AdvancedStacker
from stable_baselines3 import PPO

import glob
import os
import time

env = rlgym.make(
        team_size=1,
        tick_skip=8,
        self_play=True,
        reward_fn=MyRewardFunction(
            team_spirit=0,
            goal_w=10,
            shot_w=5,
            save_w=5,
            demo_w=0,
            above_w=0,
            got_demoed_w=0,
            behind_ball_w=0,
            save_boost_w=0.03,
            concede_w=0,
            velocity_w=0,
            velocity_pb_w=0,
            velocity_bg_w=0.5,
            ball_touch_w=4,
            ),
        game_speed=1,
        state_setter=WallDribble(),
        terminal_conditions=[TimeoutCondition(round(250)),  # TODO lengthen later
                             GoalScoredCondition(),
                             BallTouchGroundCondition()],
        obs_builder=AdvancedStacker(6),
        action_parser=KBMAction(n_bins=3),
        )
while True:
    list_of_files = glob.glob('./models/*.zip')
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"loading model {latest_file}")
    model = PPO.load(
                f"./models/{latest_file}",
                env,
                device="cpu",
                custom_objects={"n_envs": 1},
                )
    model._last_obs = None
    print(f"Loaded model {latest_file}")
    timeout = time.time() + 60 * 15  # 15 minutes from now check for new model
    try:
        while True:
            env.reset()
            action = np.zeros([2, 8])
            obs, _, done, gameinfo = env.step(action)
            while not done:
                # Here we sample a random action. If you have an agent, you would get an action from it here.
                action = model.predict(obs, gameinfo, deterministic=True)
                next_obs, reward, done, gameinfo = env.step(action)

                obs = next_obs
            _ = obs
            _ = gameinfo
            if time.time() > timeout:
                env.close()
                break

    finally:
        env.close()
