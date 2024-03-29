#!/usr/bin/env python
############################
# This code is the first version which:
# Grabs images from the Game engine and can publish the final command to control the robot
# Next step: this should be where we use a really simple approach to track the target
############################
import rclpy
from geometry_msgs.msg import Vector3
from rclpy.node import Node

# if source install/setup.bash the msg_ws -> then
from ir_msgs.msg import Command
from std_msgs.msg import String

import os, time
import numpy as np
import random
####################################### export PYTHONPATH=/home/USERNAME/sim_ws
from src.scuba_tracking.scuba_tracking.utils.controller_utils import PID_controller, msg_processing

from src.scuba_tracking.scuba_tracking.config import config

class controller(Node):

    def __init__(self):
        super().__init__('controller')

        self.direct_command = Command()
        self.direct_command.speed = 0.0
        self.direct_command.roll = 0.0
        self.direct_command.yaw = 0.0
        self.direct_command.heave = 0.0
        self.direct_command.pitch = 0.0

        ### To gather data while exploring the space by our controller
        self.previous_state = []
        self.batch_for_RL = []
        self.exploration_mode = True
        self.path_to_gathered_data = './sampled_scenarios/'
        if not os.path.isdir(self.path_to_gathered_data):
            os.mkdir(self.path_to_gathered_data)
            self.num_of_experiments = 0
        else:
            self.num_of_experiments = len(os.listdir(self.path_to_gathered_data))
        #workbook = xlsxwriter.Workbook('scuba_tracking#1.xlsx')
        #self.excel_writer = workbook.add_worksheet()
        self.sample_counter = 0

        self.debug = True
        self.image_size = config.IMAGE_SIZE
        self.single_object_tracking = True
        # let's say it's the indirect distance to the diver
        self.BB_THRESH = config.BB_AREA_THRESHOLD
        self.target_x = None
        self.target_y = None
        self.target_area = None

        self.vision_output_subscription = self.create_subscription(
            String,
            config.GENERATED_BB_TOPIC,
            self.data_handler,
            10)

        self.pose_subscription = self.create_subscription(
            Vector3,
            config.ROBOT_POS_TOPIC,
            self.pose_callback,
            10)

        self.command_publisher = self.create_publisher(Command, config.COMMAND_TOPIC, 30)
        # This current_state_publisher is to make sure that we have the pair of (state,action)
        self.current_state_publisher = self.create_publisher(String, '/aqua/current_state', 30)
        # CONTROLLER PART
        self.controller = PID_controller()

        self.reset_recovery_variables()

    def reset_recovery_variables(self):
        # For the Recovery part
        self.lost_target_step = 0
        self.rightSideChecked = False
        self.begin_time = time.time()
        self.changing_time = 2  # 2 seconds for the beginning

    def reward_calculation(self, current_observation):
        try:
            reward = (1 / (abs(current_observation[0]) + 0.01)) + (1 / (abs(current_observation[1]) + 0.01))

        except:
            print('object lost! reward = -1')
            reward = -1
        return reward

    def convert_to_str(self, data_):
        char = ''
        for item in data_:
            char += str(item) + ','
        return char[:-1]

    def data_handler(self, msg):
        yaw_ref = 0.0
        pitch_ref = 0.0
        speed_ref = 0.0
        # just to know how msg.data looks like:
        #1#102.01816,197.34833,214.18144,264.59863#
        #num_of_objs#obj1_bb#obj2_bb#...#
        mean_of_obj_locations = msg_processing(msg)
        print('obj: ',mean_of_obj_locations)
        random_target = config.PID_RANDOM_TARGET_MODE # for exploration
        # by default let's keep the target at the center
        if not random_target:
            self.target_x = None
            self.target_y = None
            self.target_area = None
        else:
            # Otherwise, for exploration purposes
            if self.sample_counter%250 == 0 :
                self.target_x = np.random.randint(0.45*self.image_size[0], 0.55*self.image_size[0])
                self.target_y = np.random.randint(0.45*self.image_size[1], 0.55*self.image_size[1])
                self.target_area = np.random.randint(0.7*self.BB_THRESH, 0.8*self.BB_THRESH)
                print('Target updated! ',[self.target_x,self.target_y,self.target_area])

        if mean_of_obj_locations[0]>-1:
            if self.single_object_tracking:
                yaw_ref, pitch_ref, speed_ref = self.controller(mean_of_obj_locations, self.target_x, self.target_y, self.target_area)
                self.reset_recovery_variables()
            else:
                pass #TODO -> TWO scuba divers at the same time
        else:
            # The spiral search strategy
            ##yaw_ref = self.search()
            yaw_ref = 0.0
            self.lost_target_step += 1


        ## reward calculation
        reward = self.reward_calculation(mean_of_obj_locations)

        if len(self.previous_state) > 0 and self.exploration_mode:
            # To save data in an excel file
            # self.excel_writer.write(self.sample_counter, 0, self.convert_to_str(self.previous_state)) # self.previous_state
            #
            # self.excel_writer.write(self.sample_counter, 1, self.convert_to_str([yaw_ref, pitch_ref, speed_ref]))
            #
            # self.excel_writer.write(self.sample_counter, 2, self.convert_to_str(mean_of_obj_locations))
            #
            # self.excel_writer.write(self.sample_counter, 3, str(reward)) # to preserve the unity
            # Saving as npy
            self.batch_for_RL.append([self.previous_state, np.array(self.previous_action),
                                      mean_of_obj_locations])

            if self.sample_counter%50:
                np.save(self.path_to_gathered_data+'scenario#'+str(self.num_of_experiments), self.batch_for_RL)

            # random exploration 20% chance
            if np.random.rand()>0.6:
                if np.random.rand()>0.5:
                    yaw_ref = random.uniform(0.5, 1)
                else:
                    yaw_ref = -random.uniform(0.5, 1)

            if np.random.rand() > 0.6:
                if np.random.rand()>0.5:
                    pitch_ref = random.uniform(0.1, 0.5)
                else:
                    pitch_ref = -random.uniform(0.1, 0.5)

            self.sample_counter += 1

        self.direct_command.yaw = yaw_ref
        self.direct_command.pitch = pitch_ref
        self.direct_command.speed = speed_ref
        self.direct_command.roll = 0.0
        if self.debug:
            print('speed ref: ', speed_ref)

        self.command_publisher.publish(self.direct_command)
        self.current_state_publisher.publish(msg)

        self.last_time = time.time()  # to have a view of the sampling rate
        self.previous_state = mean_of_obj_locations
        self.previous_action = [yaw_ref, pitch_ref, speed_ref]
    # for log purposes
    def pose_callback(self, msg):
        x, y, z = msg.x, msg.y, msg.z

        #print(x, y, z)
    def search(self): # return yaw rate - SPIRAL SEARCH
        # Change the view direction every so often (self.changing_time)
        if time.time() - self.begin_time>self.changing_time:
            self.begin_time = time.time()
            # Note, *2 is not enough as you have to also take into account the arrival time to the origin
            if self.lost_target_step:
                self.changing_time = self.changing_time + 2*self.changing_time
            else:
                self.changing_time *= 2
            self.rightSideChecked = not self.rightSideChecked

        if self.rightSideChecked:
            # right direction
            return 0.5
        else:
            # left direction
            return -0.5

def main():

    rclpy.init()

    node_ = controller()

    rclpy.spin(node_)

    node_.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
