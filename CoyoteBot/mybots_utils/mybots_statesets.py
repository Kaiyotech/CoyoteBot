from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM, CEILING_Z, GOAL_HEIGHT,\
    SIDE_WALL_X, BACK_WALL_Y, CAR_MAX_SPEED, CAR_MAX_ANG_VEL, BALL_RADIUS
from rlgym_tools.extra_state_setters.augment_setter import AugmentSetter
import numpy as np
from numpy import random as rand
import math

DEG_TO_RAD = 3.14159265 / 180


class WallDribble(StateSetter):
    def reset(self, state_wrapper: StateWrapper):
        rng = np.random.default_rng()
        # Set up our desired spawn location and orientation for car0 - special one on wall
        # don't at me about the magic numbers, just go with it.
        # blue should aim slightly towards orange goal, we're always blue?
        car_0 = state_wrapper.cars[0]
        x_choice = rand.choice([0, 2]) - 1
        rand_x = x_choice * (SIDE_WALL_X - 17)
        rand_y = rng.uniform(-BACK_WALL_Y + 1300, BACK_WALL_Y - 1300)
        rand_z = rng.uniform(325, CEILING_Z - 1400)
        desired_car_pos = [rand_x, rand_y, rand_z]  # x, y, z
        desired_pitch = (90 + (rng.uniform(-20, -5))) * DEG_TO_RAD
        desired_yaw = 90 * DEG_TO_RAD
        desired_roll = 90 * x_choice * DEG_TO_RAD
        desired_rotation = [desired_pitch, desired_yaw, desired_roll]

        car_0.set_pos(*desired_car_pos)
        car_0.set_rot(*desired_rotation)
        car_0.boost = 100

        car_0.set_lin_vel(0, 200 * x_choice, rng.uniform(1375, 1425))
        car_0.set_ang_vel(0, 0, 0)

        # Now we will spawn the ball in front of the car_0 with slightly less speed
        # 17 removes the change to move the car to the proper place, so middle of ball is at wall then we move it
        ball_x: np.float32
        if rand_x < 0:
            ball_x = rand_x - 17 + BALL_RADIUS
        else:
            ball_x = rand_x + 17 - BALL_RADIUS
        state_wrapper.ball.set_pos(x=ball_x, y=rand_y + (rng.uniform(20, 60)), z=rand_z + rng.uniform(150, 200))
        state_wrapper.ball.set_lin_vel(0, 200, rng.uniform(1200, 1300))
        state_wrapper.ball.set_ang_vel(0, 0, 0)

        # Loop over every car in the game, skipping 1 since we already did it
        for car in state_wrapper.cars:
            if car.id == 1:
                continue

            # shamelessly stolen from the arbitrary state setter
            car.set_pos(rng.uniform(-1472, 1472), rng.uniform(-1984, 1984), 0)
            car.set_rot(0, rng.uniform(-180, 180) * (np.pi / 180), 0)
            car.boost = 0.33

        if rand.choice([0, 1]):
            AugmentSetter.switch_teams(state_wrapper)

        if rand.choice([0, 1]):
            AugmentSetter.shuffle_players(state_wrapper)

