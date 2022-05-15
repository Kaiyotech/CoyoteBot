import rlgym
from CoyoteBot.mybots_utils.mybots_rewards import *
from CoyoteBot.mybots_utils.mybots_statesets import *
from CoyoteBot.mybots_utils.mybots_terminals import *
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_tools.extra_state_setters.augment_setter import AugmentSetter
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter
from rlgym_tools.extra_rewards.anneal_rewards import AnnealRewards
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import LiuDistanceBallToGoalReward, VelocityBallToGoalReward
from rlgym.utils.reward_functions.default_reward import DefaultReward
import numpy as np

import time


def anneal_rewards_fn():  # TODO this is throwing an error

    max_steps = 50_000_000  # TODO tune this some
    # when annealing, change the weights between 1 and 2, 2 is new
    reward1 = MyOldRewardFunction(
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
    )
    reward2 = MyRewardFunction(
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
        ball_touch_w=1,
    )

    alternating_rewards_steps = [reward1, max_steps, reward2]

    return AnnealRewards(*alternating_rewards_steps, mode=AnnealRewards.STEP)


env = rlgym.make(
        reward_fn=DoubleTapReward(),
        game_speed=100,
        state_setter=WeightedSampleSetter((
                        TestScore(),
                        AugmentSetter(
                            BallFrontGoalState(),
                            shuffle_within_teams=True,
                            swap_front_back=False,
                            swap_left_right=False,
                            ),
                        ),
                        (
                        0,
                        1,
                        ),
                    ),
        terminal_conditions=[TimeoutCondition(200), GoalScoredCondition()],
        self_play=True,
        )
try:
    total_time = 0
    n = 0
    while n < 1000:
        env.reset()
        done = False
        printed = False
        previous_speed = -1
        steps = 0
        ep_reward = 0
        t0 = time.time()
        while not done:
            # Here we sample a random action. If you have an agent, you would get an action from it here.
            # action = env.action_space.sample()
            action = [1, 0, 0, 0, 0, 0, 0, 0] * 2

            next_obs, reward, done, gameinfo = env.step(action)
            # ep_reward += reward
            steps += 1
            state = gameinfo["state"]
            result = gameinfo["result"]
            goal_speed = np.linalg.norm(state.ball.linear_velocity) * 0.036  # kph

            if any(reward) > 0:
                # print(reward)
                pass

            if result != 0 and not printed:
                # print(f"goal speed: {previous_speed:.2f}")
                result = 0
                printed = True

            previous_speed = goal_speed

            obs = next_obs

        length = time.time() - t0
        # print("Step time: {:1.5f} | Episode time: {:.2f} | Episode Reward: {:.2f}".format(length / steps, length,
                                                                                        #  ep_reward))
        n += 1
        total_time += length

    print(f"{n} steps took total of {total_time:.3f} for an average of {total_time/n:.5f}")

finally:
    env.close()
