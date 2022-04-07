from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy

from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from rlgym_tools.extra_obs.advanced_stacker import AdvancedStacker
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter  # TODO add later
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym_tools.extra_rewards.anneal_rewards import AnnealRewards
from rlgym.envs import Match

from mybots_utils.mybots_rewards import MyRewardFunction, MyOldRewardFunction
from mybots_utils.mybots_statesets import *
from mybots_utils.mybots_terminals import *

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
    target_steps = 2_400_000  # steps to do per rollout
    agents_per_match = 2
    num_instances = 12
    steps = target_steps // (num_instances * agents_per_match)
    batch_size = steps // 4
    n_bins = 3
    n_epochs = 10

    print(f"time per step={t_step}, gamma={gamma})")


    def anneal_rewards_fn():

        max_steps = 20_000_000  # TODO tune this some
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
                        velocity_pb_w=0.2,
                        velocity_bg_w=0.75,
                        ball_touch_w=1,
                    )

        reward2 = MyRewardFunction(
                        team_spirit=0,
                        goal_w=10,
                        aerial_goal_w=25,
                        double_tap_goal_w=75,
                        shot_w=5,
                        save_w=20,
                        demo_w=0,
                        above_w=0,
                        got_demoed_w=0,
                        behind_ball_w=0,
                        save_boost_w=0.03,
                        concede_w=-1,
                        velocity_w=0.25,
                        velocity_pb_w=0.8,
                        velocity_bg_w=1.25,
                        ball_touch_w=1,
                    )

        alternating_rewards_steps = [reward1, max_steps, reward2]

        return AnnealRewards(*alternating_rewards_steps, mode=AnnealRewards.STEP)

    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            team_size=agents_per_match // 2,
            tick_skip=frame_skip,
            reward_function=anneal_rewards_fn(),
            self_play=True,
            terminal_conditions=[TimeoutCondition(round(20 // t_step)),  # TODO lengthen later
                                 GoalScoredCondition()],
            obs_builder=AdvancedStacker(6),
            state_setter=AugmentSetter(WallDribble(),
                                       shuffle_within_teams=True,
                                       swap_front_back=False,
                                       swap_left_right=False
                                       ),
            action_parser=KBMAction(n_bins=n_bins),
            game_speed=100,  # TODO set this back to 100 after testing

        )


    env = SB3MultipleInstanceEnv(match_func_or_matches=get_match,
                                 num_instances=num_instances,
                                 wait_time=90,
                                 )
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
            net_arch=[512, 512, dict(pi=[256, 256, 256], vf=[256, 256, 256])],
        )
        model = PPO(
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

    # Save model every so often
    # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
    # This saves to specified folder with a specified name
    callback = CheckpointCallback(round(4_800_000 / env.num_envs),
                                  save_path="models",
                                  name_prefix="rl_model",
                                  )

    while True:
        model.learn(14_400_000, callback=callback, reset_num_timesteps=False)
        model.save("models/exit_save")
        model.save(f"mmr_models/{model.num_timesteps}")
