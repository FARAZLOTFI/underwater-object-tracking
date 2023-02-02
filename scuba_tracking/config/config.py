NEWER_VERSION = True
REAL_WORLD = True

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
IMAGE_SIZE = (416, 416)
BB_AREA_THRESHOLD = 20000

PID_RANDOM_TARGET_MODE = False
SATURATED_PID = False

MIN_YAW_RATE = -0.5
MAX_YAW_RATE = 0.5
MIN_PITCH_RATE = -0.05
MAX_PITCH_RATE = 0.05


