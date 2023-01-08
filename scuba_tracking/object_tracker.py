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
# to publish the result of the processed images
from std_msgs.msg import String

import os, time
####################################### detector ######
from src.scuba_tracking.scuba_tracking.test_customized import process_single_frame
from torchsummary import summary
from src.scuba_tracking.scuba_tracking.models.vision_model import scuba_detector
import torch
import time
class object_tracker(Node):

    def __init__(self):
        super().__init__('object_tracker')

        self.DEBUG = True

        self.DNN_model = scuba_detector().cuda()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        MODEL_CHECKPOINT = '/home/faraz/sim_ws/src/scuba_tracking/scuba_tracking/trained_model/training_checkpoint'
        self.DNN_model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=device)['model_state_dict'], strict=True)

        summary(self.DNN_model, (3, 416, 416))

        self.DNN_model.eval()

        self.msg_ = String()

        self.image_subscription = self.create_subscription(
            CompressedImage,
            '/simulator/front_left_camera',
            self.image_handler,
            30)

        self.data_publisher = self.create_publisher(String, '/aqua/detected_objects', 30)

        self.recording_flag = False

        if self.recording_flag:
            self.num_of_videos = 0
            self.path_to_recorded_video = './resulted_tracking/'
            if not os.path.isdir(self.path_to_recorded_video):
                os.mkdir(self.path_to_recorded_video)
            else:
                self.num_of_videos = len(os.listdir(self.path_to_recorded_video))

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            frame_size = (400, 300)
            self.out = cv2.VideoWriter('tacking_scenario_'+str(self.num_of_videos)+'.avi', fourcc, 30.0, frame_size)

        cv2.namedWindow("Front view cam", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Front view cam", 500, 400)

    def image_handler(self, msg):
        last_time = time.time()
        subscribed_image = CvBridge().compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

        string_output, outputs, img_ = process_single_frame(self.DNN_model, subscribed_image)
        # data frame: num_of_obj, obj1, obj2, obj3, ...
        self.msg_.data = str(len(outputs)) + string_output # sth like: 3#120,200#100,205#50,200# ->[1:-1]

        self.data_publisher.publish(self.msg_)
        print(time.time() - last_time)
        cv2.imshow("Front view cam", img_)
        key_ = cv2.waitKey(1)
        if self.recording_flag:
            self.out.write(img_)

def main():

    rclpy.init()

    node_ = object_tracker()

    rclpy.spin(node_)

    node_.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
