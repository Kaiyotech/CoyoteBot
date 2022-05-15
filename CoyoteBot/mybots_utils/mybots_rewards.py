import numpy as np

from rlgym.utils import math
from rlgym.utils.common_values import BLUE_TEAM, BLUE_GOAL_BACK, ORANGE_GOAL_BACK, ORANGE_TEAM, BALL_MAX_SPEED, \
    CAR_MAX_SPEED, BALL_RADIUS, GOAL_HEIGHT
from rlgym.utils.gamestates import GameState, PlayerData

from rlgym.utils.math import cosine_similarity

from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions.common_rewards.misc_rewards import *
from rlgym.utils.reward_functions.common_rewards.conditional_rewards import *
from rlgym.utils.reward_functions import RewardFunction, CombinedReward
from rlgym_tools.extra_rewards.distribute_rewards import DistributeRewards


class MyOldRewardFunction(CombinedReward):
    def __init__(
            self,
            team_spirit=0.2,
            goal_w=10,
            aerial_goal_w=25,
            double_tap_goal_w=75,
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
            ball_touch_w=1,
    ):
        self.team_spirit = team_spirit
        self.goal_w = goal_w
        self.aerial_goal_w = aerial_goal_w
        self.double_tap_goal_w = double_tap_goal_w
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
        self.ball_touch_w = ball_touch_w
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
                AerialRewardPerTouch(exp_base=1.08, max_touches_reward=50),
                AerialGoalReward(),
                DoubleTapReward(),
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
                self.ball_touch_w,
                self.aerial_goal_w,
                self.double_tap_goal_w,
            )
        )


class MyRewardFunction(CombinedReward):
    def __init__(
            self,
            team_spirit=0.2,
            goal_w=10,
            aerial_goal_w=25,
            double_tap_goal_w=75,
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
            ball_touch_w=1,
    ):
        self.team_spirit = team_spirit
        self.goal_w = goal_w
        self.aerial_goal_w = aerial_goal_w
        self.double_tap_goal_w = double_tap_goal_w
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
        self.ball_touch_w = ball_touch_w
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
                AerialRewardPerTouch(exp_base=1.75, max_touches_reward=50),
                AerialGoalReward(),
                DoubleTapReward(),
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
                self.ball_touch_w,
                self.aerial_goal_w,
                self.double_tap_goal_w,
            )
        )


class AerialGoalReward(RewardFunction):

    def __init__(self):
        self.last_touch_height = 0
        self.last_goal = 0

    def reset(self, initial_state: GameState):
        self.last_touch_height = 0
        self.last_goal = initial_state.blue_score

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if state.last_touch == player.car_id:
            self.last_touch_height = player.car_data.position[2]

        if self.last_touch_height > GOAL_HEIGHT and state.blue_score > self.last_goal:
            return 1

        else:
            return 0


class NonReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0


class UglyLandingPunish(RewardFunction):

    def __init__(self):
        pass

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        pass


class DoubleTapReward(RewardFunction):

    def __init__(self):
        self.backboard_bounce = False
        self.floor_bounce = False
        self.last_goal = 0
        self.last_ball_vel_y = 0

    def reset(self, initial_state: GameState):
        self.last_goal = initial_state.blue_score
        self.backboard_bounce = False
        self.floor_bounce = False
        self.last_ball_vel_y = initial_state.ball.linear_velocity[1]

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if state.ball.position[2] < BALL_RADIUS * 2 and state.last_touch != player.car_id:
            self.floor_bounce = True

        elif 0.55 * self.last_ball_vel_y < state.ball.linear_velocity[1] > 0.65 * self.last_ball_vel_y and\
                state.ball.position[1] > 4900 and state.ball.position[2] > 500:
            self.backboard_bounce = True

        if state.blue_score > self.last_goal and self.backboard_bounce and not self.floor_bounce:
            return 1

        else:
            return 0


class AerialRewardPerTouch(RewardFunction):
    """
        Rewards consecutive touches in the air
        :param exp_base: exp_base^n where n is consecutive touches
        :param max_touches_reward: maximum reward to give regardless of consecutive touches
        """

    def __init__(self, exp_base=1.06, max_touches_reward=20):
        self.exp_base = exp_base
        self.max_touches_reward = max_touches_reward
        self.num_touches = 0

    def reset(self, initial_state: GameState):
        self.num_touches = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if state.last_touch == player.car_id and state.ball.position[2] > 300:
            self.num_touches += 1
            reward = self.exp_base ** self.num_touches
            return min(reward, self.max_touches_reward)

        elif state.ball.position[2] <= 300:
            self.num_touches = 0
            return 0

        else:
            return 0


class IncreaseRewardPerTouch(RewardFunction):
    """
    Rewards consecutive touches
    :param exp_base: exp_base^n where n is consecutive touches
    :param max_touches_reward: maximum reward to give regardless of consecutive touches
    """

    def __init__(self, exp_base=1.06, max_touches_reward=20):
        self.exp_base = exp_base
        self.max_touches_reward = max_touches_reward
        self.num_touches = 0

    def reset(self, initial_state: GameState):
        self.num_touches = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if state.last_touch == player.car_id:
            self.num_touches += 1
            reward = self.exp_base ** self.num_touches
            return min(reward, self.max_touches_reward)

        else:
            self.num_touches = 0
            return 0


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
        if car_z < GOAL_HEIGHT:
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


# TODO delete this
class MyVelocityBallToGoalReward(RewardFunction):
    def __init__(self, own_goal=False, use_scalar_projection=False):
        super().__init__()
        self.own_goal = own_goal
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM and not self.own_goal \
                or player.team_num == ORANGE_TEAM and self.own_goal:
            objective = np.array(ORANGE_GOAL_BACK)
        else:
            objective = np.array(BLUE_GOAL_BACK)

        print(f"before is {state.ball.linear_velocity}")
        vel = state.ball.linear_velocity # np.zeros(3)
        print(f"velocity is {vel}")
        print(f"after is {state.ball.linear_velocity}")

        pos_diff = objective - state.ball.position
        if self.use_scalar_projection:
            # Vector version of v=d/t <=> t=d/v <=> 1/t=v/d
            # Max value should be max_speed / ball_radius = 2300 / 94 = 24.5
            # Used to guide the agent towards the ball
            inv_t = math.scalar_projection(vel, pos_diff)
            return inv_t
        else:
            # Regular component velocity
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            vel /= BALL_MAX_SPEED
            return float(np.dot(norm_pos_diff, vel))

