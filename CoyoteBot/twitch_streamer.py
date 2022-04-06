import rlgym
from mybots_utils.mybots_rewards import *
from mybots_utils.mybots_statesets import *
from mybots_utils.mybots_terminals import *
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from rlgym_tools.extra_obs.advanced_stacker import AdvancedStacker
from rlgym.envs import Match
from stable_baselines3 import PPO
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from mybots_utils.agent import Agent
from rlgym.utils.gamestates.player_data import PlayerData

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
    agent = Agent()
    game_state = GameState()
    player = PlayerData()
    action: agent.actor.action_space = None
    obs_builder = AdvancedStacker()
    act_parser = KBMAction(n_bins=3)
    tick_skip = 8
    timeout = time.time() + 60 * 15  # 15 minutes from now check for new model
    try:
        while True:
            env.reset()
            obs = obs_builder.build_obs(player, game_state, np.zeros([2, 5]))
            done = False
            while not done:
                action = act_parser.parse_actions(agent.actor(obs), game_state)
                obs, reward, done, gameinfo = env.step(action)

            if time.time() > timeout:
                env.close()
                break

    finally:
        env.close()

