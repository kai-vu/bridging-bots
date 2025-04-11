#!C:\Users\francesca\anaconda3\envs\webots_env\python.exe

from controller import Robot, Camera, Lidar
import numpy as np
import os

# Constants
TIME_STEP = 8  # Time step for updates
RECOGNITION_INTERVAL = 100  # Object recognition frequency
SAFE_DISTANCE = 1.0  # Minimum distance to avoid obstacles (meters)
TURN_SPEED = 2.0  # Turning speed when obstacle detected
FORWARD_SPEED = 3.0  # Forward speed when no obstacles
OBJECT_FILE = "object_positions.txt"  # File to store object positions

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
left_motor = robot.getDevice("wheel_left_joint")
right_motor = robot.getDevice("wheel_right_joint")

# Ensure motors are in velocity mode
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Object storage
step_counter = 0  # Counter to track when to run recognition again


def load_existing_objects():
    """Loads existing object positions from the file to avoid duplicates."""
    if not os.path.exists(OBJECT_FILE):
        return set()

    existing_objects = set()
    with open(OBJECT_FILE, "r") as file:
        for line in file:
            existing_objects.add(line.strip())  # Store existing objects
    return existing_objects


def recognize_objects(existing_objects):
    """Detects objects and writes unique positions to a file."""
    objects = eurobin_camera.getRecognitionObjects()
    
    if not objects:
        print("No objects detected.")
        return existing_objects
    
    with open(OBJECT_FILE, "a") as file:  # Open file in append mode
        for obj in objects:
            obj_name = obj.getModel()
            obj_position = list(obj.getPosition())  # Convert to a readable list
            position_str = f"{obj_name}: {obj_position}"

            if position_str not in existing_objects:  # Only add new objects
                existing_objects.add(position_str)
                print(position_str)
                file.write(position_str + "\n")  # Write to file

    return existing_objects


def check_obstacles():
    """Uses LiDAR to check if obstacles are nearby."""
    lidar_data = lidar.getRangeImage()
    
    if lidar_data is None or len(lidar_data) == 0:
        return False  # No valid LiDAR data
    
    # Get the front-facing LiDAR distance (average of center points)
    front_distance = np.mean(lidar_data[len(lidar_data)//2 - 10 : len(lidar_data)//2 + 10])

    if front_distance < SAFE_DISTANCE:
        return True  # Obstacle detected
    return False


# Load existing object positions at the start
existing_objects = load_existing_objects()

print("Robot is recognizing objects and navigating...")
while robot.step(TIME_STEP) != -1:
    # Object recognition every RECOGNITION_INTERVAL steps
    if step_counter % RECOGNITION_INTERVAL == 0:
        existing_objects = recognize_objects(existing_objects)

    # Obstacle avoidance
    if check_obstacles():
        print("Obstacle detected! Turning...")
        left_motor.setVelocity(-TURN_SPEED)
        right_motor.setVelocity(TURN_SPEED)
    else:
        left_motor.setVelocity(FORWARD_SPEED)
        right_motor.setVelocity(FORWARD_SPEED)

    step_counter += 1

