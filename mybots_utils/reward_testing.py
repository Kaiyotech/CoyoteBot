import rlgym
from mybots_utils import mybots_rewards
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym.utils.gamestates import GameState, PlayerData

env = rlgym.make(terminal_conditions=TimeoutCondition(round((120/8) * 30)),
                 reward_fn=mybots_rewards.OnWall(),
                 game_speed=1,)
try:
    while True:
        obs = env.reset()
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
