import rlgym
from CoyoteBot.mybots_utils.mybots_rewards import *
from CoyoteBot.mybots_utils.mybots_statesets import *
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from CoyoteBot.mybots_utils.mybots_terminals import *
from rlgym.utils.state_setters import DefaultState
from rlgym_tools.extra_state_setters.wall_state import WallPracticeState

env = rlgym.make(
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
        terminal_conditions=[BallTouchGroundCondition()],
        )
try:
    while True:
        env.reset()
        done = False
        while not done:
            # Here we sample a random action. If you have an agent, you would get an action from it here.
            action = env.action_space.sample()

            next_obs, reward, done, gameinfo = env.step(action)

            if reward > 0:
                # print(reward)
                pass

            obs = next_obs

finally:
    env.close()
