import numpy as np

from rlgym.utils import math
from rlgym.utils.common_values import BLUE_TEAM, BLUE_GOAL_BACK, ORANGE_GOAL_BACK, ORANGE_TEAM, BALL_MAX_SPEED, \
    CAR_MAX_SPEED, BALL_RADIUS
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.math import cosine_similarity

APRX_CROSSBAR_HEIGHT = 640


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

