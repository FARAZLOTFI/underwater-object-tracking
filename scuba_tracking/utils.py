import os, time
import numpy as np
#######################################

class PID_controller:
    def __init__(self, reference_values = None, single_object_tracking = True):
        self.reset()
        # gains
        self.integral_filtering = 0.95
        self.area_controller_gains = [-8,-2,0]
        self.x_controller_gains = [2,1,0]
        self.y_controller_gains = [0.13,2,0]
        self.single_object_tracking = single_object_tracking
        #self.reference_values = reference_values
        self.image_size = (400,300)

        # let's say it's the indirect distance to the diver
        self.BB_THRESH = 4500


    def __call__(self,current_observation):
        self.output_update(current_observation)
        return self.x_controller_output, self.y_controller_output, self.area_controller_output

    def output_update(self, current_observation):
        # PID controller errors
        # we have three different directions
        # longitudinal
        area_P = (current_observation[2] - self.BB_THRESH) / (self.image_size[0] * self.image_size[1])
        area_D = area_P - self.previous_area_P
        area_I = self.integral_filtering * area_P + self.previous_area_P  # FIXME
        self.previous_area_P = area_P
        self.area_controller_output = self.generate_controller_output(self.area_controller_gains, [area_P, area_D, area_I])

        # lateral
        x_P = (current_observation[0] - 0.5 * self.image_size[0]) / self.image_size[0]
        x_D = x_P - self.previous_x_P
        x_I = self.integral_filtering * x_P + self.previous_x_P  # FIXME
        self.previous_x_P = x_P
        self.x_controller_output = self.generate_controller_output(self.x_controller_gains, [x_P, x_D, x_I])

        # Depth
        y_P = (current_observation[1] - 0.5 * self.image_size[1]) / self.image_size[1]
        y_D = y_P - self.previous_y_P
        y_I = self.integral_filtering * y_P + self.previous_y_P  # FIXME
        self.previous_y_P = y_P
        self.y_controller_output = self.generate_controller_output(self.y_controller_gains, [y_P, y_D, y_I])

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

    if len(data_list) > 2:
        num_of_objs = int(data_list[0])
        if single_object_tracking:
            # let's take the objs center points and the area of the BBs
            mean_of_obj_locations = np.zeros(
                3)  # this may include the center points and the areas covered by the objs
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
        return None


def main():

    controller = PID_controller()
    print('area controller gain: ',controller.area_controller_gains)
    print('x controller gain: ', controller.x_controller_gains)
    print('y controller gain: ',controller.y_controller_gains)


if __name__ == '__main__':
    main()
