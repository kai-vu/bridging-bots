from controller import Robot, Camera, RangeFinder, CameraRecognitionObject, Lidar
import math
import numpy as np

# Constants
TIME_STEP = 8  # Time step for updates
MAX_SPEED = 5  # Maximum motor speed
SAFE_DISTANCE = 2.5  # Minimum safe distance from obstacles (in meters)
TARGET_OBJECT = "fridge"  # Object to detect and approach
STOP_DISTANCE = 0.3  # Distance at which the robot stops near the object
ORIENTATION_THRESHOLD = 0.05  # More precise threshold for orientation correction
TURNING_GAIN = 3.0  # Increased gain for smoother turning
SLOW_DOWN_DISTANCE = 1.0  # Distance at which to start reducing speed
WHEELBASE = 0.4  # Distance between wheels (in meters)
AVOIDANCE_TURN_TIME = 5  # Time steps to turn when avoiding an obstacle
STUCK_THRESHOLD = 50  # Number of cycles before considering the robot stuck
BACKUP_TIME = 20  # Time steps to move backward when stuck
RECOGNITION_INTERVAL = 100  # More frequent object recognition attempts

# Initialize robot
robot = Robot()

# Enable RGB and Depth Cameras
rgb_camera = robot.getDevice("Astra rgb")
rgb_camera.enable(TIME_STEP)

depth_camera = robot.getDevice("Astra depth")
depth_camera.enable(TIME_STEP)

# Enable Eurobin Camera and Recognition
eurobin_camera = robot.getDevice("camera")
eurobin_camera.enable(TIME_STEP)
eurobin_camera.recognitionEnable(TIME_STEP)

# Enable Lidar Sensor
lidar = robot.getDevice("Hokuyo URG-04LX-UG01")
lidar.enable(TIME_STEP)
lidar.enablePointCloud()

# Allow LiDAR time to initialize
for _ in range(10):
    robot.step(TIME_STEP)

# Get motors
motor_names = [
    "head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
    "arm_2_joint", "arm_3_joint", "arm_4_joint", "arm_5_joint",
    "arm_6_joint", "arm_7_joint", "wheel_left_joint", "wheel_right_joint"
]
robot_parts = {name: robot.getDevice(name) for name in motor_names}

# Set initial positions
target_positions = [0.24, -0.67, 0.09, 0.07, 0.26, -3.16, 1.27, 1.32, 0.0, 1.41, float('inf'), float('inf')]
for name, target in zip(motor_names, target_positions):
    robot_parts[name].setPosition(target)
    robot_parts[name].setVelocity(robot_parts[name].getMaxVelocity() / 2.0)

left_motor = robot_parts["wheel_left_joint"]
right_motor = robot_parts["wheel_right_joint"]

# Ensure motors are in velocity mode
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Keep the arm raised
robot_parts["arm_6_joint"].setPosition(1.0)
robot_parts["arm_7_joint"].setPosition(1.0)

# Object storage
target_position = None
found_target = False
robot_reached_target = False
robot_position = [0.0, 0.0, 0.0]  # x, y, theta (heading angle in radians)
stuck_counter = 0  # Counter to detect if the robot is stuck
step_counter = 0  # Counter to track when to run recognition again


def check_lidar_for_obstacles():
    """Checks LiDAR for obstacles and ensures a clear path."""
    lidar_data = lidar.getRangeImage()
    if lidar_data is None or len(lidar_data) == 0:
        return False
    front_distance = np.mean(lidar_data[len(lidar_data)//2 - 10 : len(lidar_data)//2 + 10])
    return front_distance < SAFE_DISTANCE


def recognize_objects():
    """Detects the target object and starts navigation towards it."""
    global target_position, found_target
    objects = eurobin_camera.getRecognitionObjects()
    for obj in objects:
        model_name = obj.getModel()
        if model_name == TARGET_OBJECT:
            target_position = obj.getPosition()
            found_target = True
            print(f"{TARGET_OBJECT} detected at position: {target_position}")
            navigate_to_target()
            return True
    return False


def navigate_to_target():
    """Navigates towards the target while avoiding obstacles dynamically."""
    global target_position, robot_reached_target
    while target_position and not robot_reached_target:
        if check_lidar_for_obstacles():
            left_motor.setVelocity(-MAX_SPEED / 4)
            right_motor.setVelocity(MAX_SPEED / 4)
            robot.step(TIME_STEP * 10)
            continue
        
        direction_x = target_position[0] - robot_position[0]
        direction_y = target_position[1] - robot_position[1]
        distance_to_target = math.sqrt(direction_x**2 + direction_y**2)
        angle_to_target = math.atan2(direction_y, direction_x)
        
        turn_speed = TURNING_GAIN * angle_to_target
        left_motor.setVelocity(-turn_speed)
        right_motor.setVelocity(turn_speed)
        
        if abs(angle_to_target) <= ORIENTATION_THRESHOLD:
            left_motor.setVelocity(MAX_SPEED)
            right_motor.setVelocity(MAX_SPEED)
        
        robot.step(TIME_STEP)
        if distance_to_target <= STOP_DISTANCE:
            left_motor.setVelocity(0)
            right_motor.setVelocity(0)
            robot_reached_target = True
            print("TARGET REACHED! Robot has stopped.")
            move_forward_after_reaching_target()
            return


def move_forward_after_reaching_target():
    """Moves the robot 1 meter forward after reaching the target."""
    print("Moving 1 meter forward after reaching the target...")
    distance_moved = 0.0
    while distance_moved < 4.0:
        left_motor.setVelocity(MAX_SPEED / 2)
        right_motor.setVelocity(MAX_SPEED / 2)
        robot.step(TIME_STEP)
        distance_moved += (MAX_SPEED / 2) * (TIME_STEP / 1000.0)
    left_motor.setVelocity(0)
    right_motor.setVelocity(0)
    print("Final movement completed.")


print("Robot is recognizing objects...")
while robot.step(TIME_STEP) != -1:
    if step_counter % RECOGNITION_INTERVAL == 0:
        found = recognize_objects()
        if not found:
            left_motor.setVelocity(MAX_SPEED / 2)
            right_motor.setVelocity(MAX_SPEED / 2)
    if robot_reached_target:
        print("Simulation running, but robot stopped.")
        break  # Stop robot movement but keep simulation running
    step_counter += 1
