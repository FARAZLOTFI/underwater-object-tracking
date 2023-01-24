NEWER_VERSION = False
REAL_WORLD = False

if NEWER_VERSION:
    CAMERA_TOPIC = '/fl_cam/aqua_fl_cam/image_raw/compressed'
    ROBOT_POS_TOPIC = '/aqua/dvl_pose_estimate'
    DOWNWARD_CAM_TOPIC = '/back_camera/image_raw/compressed'
else:
    CAMERA_TOPIC = '/simulator/front_left_camera'
    ROBOT_POS_TOPIC = '/simulator/position_ground_truth'

GENERATED_BB_TOPIC = '/aqua/detected_objects'
COMMAND_TOPIC = '/aqua/command'

if REAL_WORLD:
    YOLO_WEIGHTS = './src/weights/vdd_weights.pt'
else:
    YOLO_WEIGHTS = './src/weights/simulator_weights.pt'

RL_CHECKPOINT = './RL_checkpoint/training_checkpoint'
