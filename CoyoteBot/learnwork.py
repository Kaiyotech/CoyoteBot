from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy

from rlgym.utils.action_parsers import DiscreteAction
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym_tools.extra_rewards.anneal_rewards import AnnealRewards
from rlgym.envs import Match

from mybots_utils.mybots_rewards import MyRewardFunction

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


    def anneal_rewards_fn():

        max_steps = 50_000_000  # TODO tune this some
        # when annealing, change the weights between 1 and 2, 2 is new
        reward1 = MyRewardFunction(
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
                        )
        reward2 = MyRewardFunction(
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
                        )

        alternating_rewards_steps = [reward1, max_steps, reward2]

        return AnnealRewards(*alternating_rewards_steps, mode=AnnealRewards.STEP)

    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            team_size=agents_per_match // 2,
            tick_skip=frame_skip,
            reward_function=anneal_rewards_fn(),
            self_play=True,
            terminal_conditions=[TimeoutCondition(round(30 // t_step)), GoalScoredCondition()],  # TODO lengthen later
            obs_builder=AdvancedObs(),
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
    callback = CheckpointCallback(round(18_000_000 / env.num_envs),
                                  save_path="models",
                                  name_prefix="rl_model",
                                  )

    while True:
        model.learn(36_000_000, callback=callback, reset_num_timesteps=False)
        model.save("models/exit_save")
        model.save(f"mmr_models/{model.num_timesteps}")
