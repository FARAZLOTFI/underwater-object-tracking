#!/usr/bin/env python
############################
# This code is the first version which:
# Grabs images from the Game engine and can publish the final command to control the robot
# Next step: this should be where we use a really simple approach to track the target
############################
import rclpy
from geometry_msgs.msg import Vector3
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
import cv2
from rclpy.node import Node

# if source install/setup.bash the msg_ws -> then
from ir_msgs.msg import Command

import os, time
#######################################

class dataset_collector(Node):

    def __init__(self):
        super().__init__('dataset_collector')

        self.direct_command = Command()
        self.DEBUG = True

        self.image_subscription = self.create_subscription(
            CompressedImage,
            '/simulator/front_left_camera',
            self.image_handler,
            30)

        self.pose_subscription = self.create_subscription(
            Vector3,
            '/simulator/position_ground_truth',
            self.pose_callback,
            30)

        self.command_publisher = self.create_publisher(Command, '/aqua/command', 30)

        self.num_of_samples = 0
        self.path_to_gathered_data = './sampled_images/'
        if not os.path.isdir(self.path_to_gathered_data):
            os.mkdir(self.path_to_gathered_data)
        else:
            self.num_of_samples = len(os.listdir(self.path_to_gathered_data))

        cv2.namedWindow("Front view cam", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Front view cam", 500, 400)

    def image_handler(self, msg):
        global key_

        subscribed_image = CvBridge().compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

        self.direct_command.speed = 0.0
        self.direct_command.roll = 0.0
        self.direct_command.yaw = 0.0
        self.direct_command.heave = 0.0
        self.direct_command.pitch = 0.0

        cv2.imshow("Front view cam", subscribed_image)
        key_ = cv2.waitKey(1)
        if key_ == 150: # heading left
            self.direct_command.yaw = 0.5
        elif key_ == 152: # heading right
            self.direct_command.yaw = -0.5

        self.direct_command.pitch = 0.5
        self.direct_command.yaw = 0.5
        if key_ == 151:  # arrow key ^
            self.direct_command.pitch = -0.5
        elif key_ == 153:  #
            self.direct_command.pitch = 0.5

        if key_ == ord('w'):  # go forward
            self.direct_command.speed = 0.5
        elif key_ == ord('s'): # go backward
            self.direct_command.speed = -0.5

        if key_ == ord('a'):  # go forward
            self.direct_command.roll = -0.5
        elif key_ == ord('d'): # go backward
            self.direct_command.roll = 0.5

        if key_ == 82:  # arrow key ^
            self.direct_command.heave = 0.2
        elif key_ == 84:  #
            self.direct_command.heave = -0.2

        #elif key_ == :

        if key_ == 32: # take the photo
            cv2.imwrite(self.path_to_gathered_data + str(self.num_of_samples) + '.jpg', subscribed_image)
            print('Captured a photo, we have '+str(self.num_of_samples)+' number of samples now.')
            self.num_of_samples +=1
            time.sleep(0.3)


    def pose_callback(self, msg):
        global key_
        x, y, z = msg.x, msg.y, msg.z

        self.command_publisher.publish(self.direct_command)
        #print(x, y, z)


def main():

    rclpy.init()

    node_ = dataset_collector()

    rclpy.spin(node_)

    node_.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
