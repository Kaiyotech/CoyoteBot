import numpy as np

from rlgym.utils import math
from rlgym.utils.common_values import BLUE_TEAM, BLUE_GOAL_BACK, ORANGE_GOAL_BACK, ORANGE_TEAM, BALL_MAX_SPEED, \
    CAR_MAX_SPEED, BALL_RADIUS
from rlgym.utils.gamestates import GameState, PlayerData

from rlgym.utils.math import cosine_similarity

from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions.common_rewards.misc_rewards import *
from rlgym.utils.reward_functions import RewardFunction, CombinedReward
from rlgym_tools.extra_rewards.distribute_rewards import DistributeRewards

APRX_CROSSBAR_HEIGHT = 640


class MyRewardFunction(CombinedReward):
    def __init__(
            self,
            team_spirit=0.2,
            goal_w=10,
            shot_w=0.2,
            save_w=5,
            demo_w=5,
            above_w=0.05,
            got_demoed_w=-6,
            behind_ball_w=0.01,
            save_boost_w=0.03,
            concede_w=-5,
            velocity_w=0.8,
            velocity_pb_w=0.5,
            velocity_bg_w=0.6,
    ):
        self.team_spirit = team_spirit
        self.goal_w = goal_w
        self.shot_w = shot_w
        self.save_w = save_w
        self.demo_w = demo_w
        self.above_w = above_w
        self.got_demoed_w = got_demoed_w
        self.behind_ball_w = behind_ball_w
        self.save_boost_w = save_boost_w
        self.concede_w = concede_w
        self.velocity_w = velocity_w
        self.velocity_pb_w = velocity_pb_w
        self.velocity_bg_w = velocity_bg_w
        # self.rewards = None
        goal_reward = EventReward(goal=self.goal_w, concede=self.concede_w)
        distrib_reward = DistributeRewards(goal_reward, team_spirit=self.team_spirit)
        super().__init__(
            reward_functions=(
                distrib_reward,
                AboveCrossbar(),
                SaveBoostReward(),
                VelocityReward(),
                VelocityPlayerToBallReward(),
                VelocityBallToGoalReward(),
                Demoed(),
                EventReward(
                    shot=self.shot_w,
                    save=self.save_w,
                    demo=self.demo_w,
                ),
            ),
            reward_weights=(
                1.0,
                self.above_w,
                self.save_boost_w,
                self.velocity_w,
                self.velocity_pb_w,
                self.velocity_bg_w,
                self.got_demoed_w,
                1.0,
            )
        )


class AboveCrossbar(RewardFunction):
    # reward function if above the crossbar and near the ball
    # or aimed towards
    # ball, to reward good aerials pretty aggressively
    def __init__(self, defense=1., offense=1.):
        self.defense = defense
        self.offense = offense

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ball_pos = state.ball.position
        pos = player.car_data.position
        # not above crossbar
        car_z = pos[2]
        if car_z < APRX_CROSSBAR_HEIGHT:
            return 0

        # from player_ball_rewards rlgym
        dist = np.linalg.norm(player.car_data.position - state.ball.position) - BALL_RADIUS
        close = np.exp(-0.5 * dist / CAR_MAX_SPEED)  # Inspired by https://arxiv.org/abs/2105.12196

        # Align player->ball and player->net vectors
        alignment = 0.5 * (cosine_similarity(ball_pos - pos, ORANGE_GOAL_BACK - pos)
                           - cosine_similarity(ball_pos - pos, BLUE_GOAL_BACK - pos))
        if player.team_num == ORANGE_TEAM:
            alignment *= -1

        return close + alignment


class OnWall(RewardFunction):

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        on_ground = player.on_ground
        absz = abs(player.car_data.up()[2])
        reward = 0
        pos = player.car_data.position
        car_x = pos[0]
        car_y = pos[1]
        car_z = pos[2]
        if on_ground and absz < 0.2:
            reward = 1
        elif on_ground and absz < 0.9:
            reward = 0.2
        return reward


class Demoed(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.is_demoed:
            return 1
        else:
            return 0

