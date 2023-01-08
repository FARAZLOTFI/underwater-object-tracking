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
####################################### export PYTHONPATH=/home/USERNAME/sim_ws
from src.scuba_tracking.scuba_tracking.utils import PID_controller, msg_processing

class controller(Node):

    def __init__(self):
        super().__init__('controller')

        self.direct_command = Command()
        self.direct_command.speed = 0.0
        self.direct_command.roll = 0.0
        self.direct_command.yaw = 0.0
        self.direct_command.heave = 0.0
        self.direct_command.pitch = 0.0

        self.DEBUG = True
        self.image_size = (400,300)

        # let's say it's the indirect distance to the diver
        self.BB_THRESH = 4800

        self.vision_output_subscription = self.create_subscription(
            String,
            '/aqua/detected_objects',
            self.data_handler,
            30)

        self.pose_subscription = self.create_subscription(
            Vector3,
            '/simulator/position_ground_truth',
            self.pose_callback,
            30)

        self.command_publisher = self.create_publisher(Command, '/aqua/command', 30)

        # CONTROLLER PART
        self.controller = PID_controller()

    def data_handler(self, msg):
        # just to know how msg.data looks like:
        #1#102.01816,197.34833,214.18144,264.59863#
        #num_of_objs#obj1_bb#obj2_bb#...#
        # check the validity of the coming data (at least three "#" should exist in the message):
        data_list = msg.data.split('#')
        print('data: ',len(data_list), int(data_list[0]))
        if len(data_list)>2:
            num_of_objs = int(data_list[0])
            # let's take the objs center points and the area of the BBs
            mean_of_obj_locations = np.zeros(3) # this may include the center points and the areas covered by the objs
            for i in range(num_of_objs):
                print('hey: ',data_list[i + 1])
                x1, y1, x2, y2 = list(map(float, data_list[i+1].split(','))) # just to ignore the first element as is the num of objs
                mean_of_obj_locations[0] += (x1 + x2)/ (2*num_of_objs)
                mean_of_obj_locations[1] += (y1 + y2) / (2*num_of_objs)
                mean_of_obj_locations[2] += (y2 - y1)*(x2 - x1)/num_of_objs

            yaw_ref, pitch_ref, speed_ref = self.controller(mean_of_obj_locations)

            # self.direct_command.yaw = yaw_ref
            # self.direct_command.pitch = pitch_ref
            # self.direct_command.speed = speed_ref
            # self.direct_command.roll = 0.0
            # print('speed ref: ', speed_ref)

        else:
            # NO OBJ/ OBJ lost!
            # Here we may implement the part relative to recovery, etc
            pass

        self.command_publisher.publish(self.direct_command)
    # for log purposes
    def pose_callback(self, msg):
        global key_
        x, y, z = msg.x, msg.y, msg.z

        #print(x, y, z)

def main():

    rclpy.init()

    node_ = controller()

    rclpy.spin(node_)

    node_.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
