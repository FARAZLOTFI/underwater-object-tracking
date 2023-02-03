#!/usr/bin/env python
############################
# This code is the first version which:
# Grabs images from the Game engine and can publish the final command to control the robot
# Next step: this should be where we use a really simple approach to track the target
############################
import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
import cv2
from rclpy.node import Node
from src.scuba_tracking.scuba_tracking.models.yolov7 import YoloV7
from std_msgs.msg import String
import os, time

from src.scuba_tracking.scuba_tracking.config import config
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy


class object_tracker(Node):

    def __init__(self):
        super().__init__('object_tracker')

        print('cam topic: ',config.CAMERA_TOPIC)
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
        )
        self.image_subscription = self.create_subscription(
            CompressedImage,
            config.CAMERA_TOPIC,
            self.image_handler,
            qos_profile=qos_profile)
        self.data_publisher = self.create_publisher(String, config.GENERATED_BB_TOPIC, 10)
        self.msg_ = String()
        self.recording_flag = False
        if self.recording_flag:
            self.num_of_videos = 0
            self.path_to_recorded_video = './resulted_tracking/'
            if not os.path.isdir(self.path_to_recorded_video):
                os.mkdir(self.path_to_recorded_video)
            else:
                self.num_of_videos = len(os.listdir(self.path_to_recorded_video))

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            frame_size = config.IMAGE_SIZE
            self.out = cv2.VideoWriter('tracking_scenario_'+str(self.num_of_videos)+'.avi', fourcc, 30.0, frame_size)

        self.detector = YoloV7()
        cv2.namedWindow("Processed frames", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Processed frames", config.IMAGE_SIZE[0], config.IMAGE_SIZE[1])

        return

    def image_handler(self, msg):
        img = CvBridge().compressed_imgmsg_to_cv2(msg)
        last_time = time.time()
        img = cv2.resize(img, config.IMAGE_SIZE)
        string_output, outputs, img_ = self.detector.detect(img)
        string_command = str(len(outputs)) + string_output
        print(time.time() - last_time)
        cv2.imshow("Processed frames", img_)
        key_ = cv2.waitKey(1)

        self.msg_.data = string_command
        self.data_publisher.publish(self.msg_)
        if self.recording_flag:
            self.out.write(img_)




def main(args=None):
    rclpy.init()

    object_tracker_ = object_tracker()

    rclpy.spin(object_tracker_)

    object_tracker_.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
