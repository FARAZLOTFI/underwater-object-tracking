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

class object_tracker(Node):

    def __init__(self):
        super().__init__('object_tracker')
        self.detector = YoloV7()
        self.image_subscription = self.create_subscription(
            CompressedImage,
            '/simulator/front_left_camera',
            self.image_handler,
            30)
        self.data_publisher = self.create_publisher(String, '/aqua/detected_objects', 30)
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
            frame_size = (400, 300)
            self.out = cv2.VideoWriter('tracking_scenario_'+str(self.num_of_videos)+'.avi', fourcc, 30.0, frame_size)
        cv2.namedWindow("Front view cam", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Front view cam", 500, 400)
        return

    def image_handler(self, msg):
        last_time = time.time()
        img = CvBridge().compressed_imgmsg_to_cv2(msg)
        string_output, outputs, img_ = self.detector.detect(img)
        self.msg_.data = str(len(outputs)) + string_output
        self.data_publisher.publish(self.msg_)
        print(time.time() - last_time)
        cv2.imshow("Front view cam", img_)
        cv2.waitKey(1)  
        if self.recording_flag:
            self.out.write(img_)
        return

def main(args=None):
    rclpy.init(args=args)

    object_tracker_ = object_tracker()

    rclpy.spin(object_tracker_)

    object_tracker_.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
