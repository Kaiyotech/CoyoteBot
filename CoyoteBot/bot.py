from rlgym.envs import Match
from rlgym.utils.action_parsers import DiscreteAction
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy

from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym.utils.reward_functions.common_rewards.misc_rewards import *
from rlgym.utils.reward_functions.common_rewards.misc_rewards import SaveBoostReward
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
# from rlgym.utils.reward_functions.common_rewards.conditional_rewards import RewardIfBehindBall
from rlgym.utils.reward_functions import CombinedReward
from mybots_utils.mybots_rewards import *

from os.path import exists

from torch.nn import ReLU

if __name__ == '__main__':  # Required for multiprocessing
    frame_skip = 8  # Number of ticks to repeat an action
    # TODO maybe the action can instead be as an array [start,stop] from [0,7] or just [2,2]
    #   or just speed it up maybe?
    #
    time_horizon = 15  # horizon in seconds
    t_step = frame_skip / 120  # real game time per rollout step

    gamma = 1 - (t_step / time_horizon)
    gae_lambda = 0.95
    learning_rate = 5e-5
    ent_coef = 0.01
    vf_coef = 1.
    target_steps = 6_000_000  # steps to do per rollout
    agents_per_match = 6
    num_instances = 10
    steps = target_steps // (num_instances * agents_per_match)
    batch_size = steps // 5
    n_bins = 101
    n_epochs = 10

    print(f"time per step={t_step}, gamma={gamma})")


    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            team_size=agents_per_match // 2,
            tick_skip=frame_skip,
            reward_function=CombinedReward(
                (
                    # TODO anneal when adding complexity
                    # TODO add team spirit
                    # AboveCrossbar(),
                    # OnWall(),
                    # SaveBoostReward(),
                    # RewardIfBehindBall(),
                    VelocityReward(),
                    VelocityPlayerToBallReward(),
                    VelocityBallToGoalReward(),
                    # Demoed(),
                    EventReward(
                        team_goal=100.0,
                        concede=0,  # add later
                        shot=20.0,
                        save=0,  # add later
                        demo=0,  # add later
                    ),
                ),
                (0.8, 0.5, 0.6, 1.0)
            ),
            self_play=True,
            terminal_conditions=[TimeoutCondition(round(30 // t_step)), GoalScoredCondition()],  # TODO lengthen later
            obs_builder=AdvancedObs(),  # Not that advanced, good default
            state_setter=DefaultState(),  # Resets to kickoff position
            action_parser=DiscreteAction(n_bins=n_bins)
        )


    env = SB3MultipleInstanceEnv(match_func_or_matches=get_match,
                                 num_instances=num_instances)
    env = VecCheckNan(env)  # Optional
    env = VecMonitor(env)  # Recommended, logs mean reward and ep_len to Tensorboard
    env = VecNormalize(env, norm_obs=False, gamma=gamma)  # Highly recommended, normalizes rewards
    if exists("models/exit_save.zip"):
        model = PPO.load(
            "models/exit_save.zip",
            env,
            device="auto",
            custom_objects={"n_envs": env.num_envs},
            )
        model._last_obs = None
        print("Loaded previous exit_save model")
    else:
        policy_kwargs = dict(
            activation_fn=ReLU,
            net_arch=[512, dict(pi=[256, 256], vf=[512, 512])],
        )
        model = PPO( # TODO initialize with zero mean, low deviation, xavier?
            MlpPolicy,
            env,
            n_epochs=n_epochs,
            policy_kwargs=policy_kwargs,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            gamma=gamma,
            gae_lambda=gae_lambda,
            verbose=3,  # Print out all the info as we're going
            batch_size=batch_size,  # Batch size as high as possible within reason
            n_steps=steps,  # Number of steps per env to perform before optimizing network
            tensorboard_log="logs",  # `tensorboard --logdir out/logs` in terminal to see graphs
            device="auto"  # Uses GPU if available
        )
        # TODO figure out how to initialize

    # Save model every so often
    # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
    # This saves to specified folder with a specified name
    callback = CheckpointCallback(round(18_000_000 / env.num_envs),
                                  save_path="models",
                                  name_prefix="rl_model",
                                  )

    while True:
        model.learn(36_000_000, callback=callback, reset_num_timesteps=False)
        model.save("models/exit_save")
        model.save(f"mmr_models/{model.num_timesteps}")
