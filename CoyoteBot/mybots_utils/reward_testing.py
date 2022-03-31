import rlgym
from CoyoteBot.mybots_utils import mybots_rewards
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition

env = rlgym.make(
        reward_fn=mybots_rewards.MyRewardFunction(
            team_spirit=0.3,
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
            ),
        game_speed=1, )
try:
    while True:
        env.reset()
        done = False
        while not done:
            # Here we sample a random action. If you have an agent, you would get an action from it here.
            action = env.action_space.sample()

            next_obs, reward, done, gameinfo = env.step(action)
            if reward > 0:
                print(reward)

            obs = next_obs

finally:
    env.close()
