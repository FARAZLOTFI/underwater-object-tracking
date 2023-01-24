NEWER_VERSION = False

if NEWER_VERSION:
    CAMERA_TOPIC = '/fl_cam/aqua_fl_cam/image_raw/compressed'
    ROBOT_POS_TOPIC = '/aqua/dvl_pose_estimate'
    DOWNWARD_CAM_TOPIC = '/back_camera/image_raw/compressed'
else:
    CAMERA_TOPIC = '/simulator/front_left_camera'
    ROBOT_POS_TOPIC = '/simulator/position_ground_truth'

GENERATED_BB_TOPIC = '/aqua/detected_objects'
COMMAND_TOPIC = '/aqua/command'

YOLO_WEIGHTS = '/home/faraz/sim_ws/src/scuba_tracking/weights/simulator_weights.pt'

