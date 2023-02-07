import os, time
import numpy as np
from src.scuba_tracking.scuba_tracking.config import config
#######################################

class PID_controller:
    def __init__(self, reference_values = None, single_object_tracking = True):
        self.reset()
        # gains
        self.integral_filtering = 0.95
        self.area_controller_gains = [-8,-2,0] #[-8,-2,0]
        self.x_controller_gains = [2,1,0]#[2,1,0]
        self.y_controller_gains = [0.5,1,0]#[0.13,2,0]
        self.single_object_tracking = single_object_tracking
        #self.reference_values = reference_values
        self.image_size = config.IMAGE_SIZE

        # let's say it's the indirect distance to the diver
        self.BB_THRESH = config.BB_AREA_THRESHOLD
        # This is critical for the safety mechanism
        self.BB_MAX = config.BB_AREA_MAX

    def __call__(self,current_observation, x = None, y = None, area = None):
        self.output_update(current_observation, x, y, area)
        return self.x_controller_output, self.y_controller_output, self.area_controller_output

    def output_update(self, current_observation, target_x, target_y, area_target):
        # PID controller errors
        # we have three different directions
        # longitudinal
        if area_target is None:
            area_target = self.BB_THRESH

        area_P = (current_observation[2] - area_target) / (self.image_size[0] * self.image_size[1])
        area_D = area_P - self.previous_area_P
        area_I = self.integral_filtering * area_P + self.previous_area_P  # FIXME
        self.previous_area_P = area_P
        self.area_controller_output = self.generate_controller_output(self.area_controller_gains, [area_P, area_D, area_I])

        # lateral
        if target_x is None:
            target_x = 0.5 * self.image_size[0]

        x_P = (current_observation[0] - target_x) / self.image_size[0]
        x_D = x_P - self.previous_x_P
        x_I = self.integral_filtering * x_P + self.previous_x_P  # FIXME
        self.previous_x_P = x_P
        self.x_controller_output = self.generate_controller_output(self.x_controller_gains, [x_P, x_D, x_I])

        # Depth'debug
        if target_y is None:
            target_y = 0.5 * self.image_size[1]

        y_P = (current_observation[1] - target_y) / self.image_size[1]
        y_D = y_P - self.previous_y_P
        y_I = self.integral_filtering * y_P + self.previous_y_P  # FIXME
        self.previous_y_P = y_P
        self.y_controller_output = self.generate_controller_output(self.y_controller_gains, [y_P, y_D, y_I])

        # SAFETY MECHANISM # @@@
        print('current observation:', current_observation[2])
        if (current_observation[2] - self.BB_MAX)>0:
            self.x_controller_output = 0.0
            self.y_controller_output = 0.0
            self.area_controller_output = 0.0
        ########################
        print('controller setpoints: ',target_x, target_y, area_target)
    def generate_controller_output(self, gains, errors):
        kp, ki, kd = gains
        error_p, error_i, error_d = errors
        return kp * error_p + ki * error_i + kd * error_d

    def reset(self):
        self.previous_area_P = 0
        self.previous_x_P = 0
        self.previous_y_P = 0

def msg_processing(msg, single_object_tracking = True):
    # check the validity of the coming data (at least three "#" should exist in the message):
    data_list = msg.data.split('#')

    mean_of_obj_locations = -np.ones(
        3)

    if len(data_list) > 2:
        num_of_objs = int(data_list[0])
        if single_object_tracking:
            # let's take the objs center points and the area of the BBs
              # this may include the center points and the areas covered by the objs
            for i in range(num_of_objs):
                # just to ignore the first element as it is the num of objs ([i + 1])
                x1, y1, x2, y2 = list(map(float, data_list[i + 1].split(',')))
                mean_of_obj_locations[0] += (x1 + x2) / (2 * num_of_objs)
                mean_of_obj_locations[1] += (y1 + y2) / (2 * num_of_objs)
                mean_of_obj_locations[2] += (y2 - y1) * (x2 - x1) / num_of_objs

            return mean_of_obj_locations
        else:
            return 0 # TODO
    else:
        return mean_of_obj_locations


def main():

    controller = PID_controller()
    print('area controller gain: ',controller.area_controller_gains)
    print('x controller gain: ', controller.x_controller_gains)
    print('y controller gain: ',controller.y_controller_gains)


if __name__ == '__main__':
    main()
