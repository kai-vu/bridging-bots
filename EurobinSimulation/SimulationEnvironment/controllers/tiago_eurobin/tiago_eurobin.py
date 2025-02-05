from controller import Robot, Keyboard, Camera, RangeFinder, CameraRecognitionObject
import math
import numpy as np
import cv2
from ultralytics import YOLO  

# Constants
TIME_STEP = 32  # time step
MAX_SPEED = 7.0  # Maximum motor speed

# Load YOLOv8 model (pretrained on COCO dataset)
model = YOLO("yolov8n.pt")  # Use 'yolov8n' for fast inference

# Initialize robot
robot = Robot()

# Enable keyboard
keyboard = Keyboard()
keyboard.enable(TIME_STEP)

# Enable RGB and Depth Cameras
rgb_camera = robot.getDevice("Astra rgb")
rgb_camera.enable(TIME_STEP)

depth_camera = robot.getDevice("Astra depth")
depth_camera.enable(TIME_STEP)

eurobin_camera = robot.getDevice("camera")
eurobin_camera.enable(TIME_STEP)
eurobin_camera.recognitionEnable(TIME_STEP)



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

# Motor control (left and right wheels)
left_motor = robot_parts["wheel_left_joint"]
right_motor = robot_parts["wheel_right_joint"]

def check_keyboard():
    """Handles keyboard input for robot movement."""
    key = keyboard.getKey()
    speeds_left, speeds_right = 0.0, 0.0

    if key == Keyboard.UP:
        speeds_left, speeds_right = MAX_SPEED, MAX_SPEED
    elif key == Keyboard.DOWN:
        speeds_left, speeds_right = -MAX_SPEED, -MAX_SPEED
    elif key == Keyboard.RIGHT:
        speeds_left, speeds_right = MAX_SPEED, -MAX_SPEED
    elif key == Keyboard.LEFT:
        speeds_left, speeds_right = -MAX_SPEED, MAX_SPEED

    left_motor.setVelocity(speeds_left)
    right_motor.setVelocity(speeds_right)

def process_camera_data():
    """Captures and processes images from the RGB camera using YOLOv8."""
    width, height = rgb_camera.getWidth(), rgb_camera.getHeight()
    image = rgb_camera.getImage()

    if image:
        # Convert Webots image to OpenCV format
        img_array = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)  # Convert BGRA to BGR

        # Run YOLOv8 object detection
        results = model(img_rgb)[0]  # Perform inference
        
        # Draw detections on image
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            label = f"{results.names[int(box.cls[0])]} {box.conf[0]:.2f}"  # Class name + confidence

            # Draw bounding box
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_rgb, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the image with detections
        cv2.imshow("Robot Camera View - YOLOv8 Detection", img_rgb)
        cv2.waitKey(1)  # Refresh image display

        # Print detected objects
        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = box.conf[0]
            object_name = results.names[class_id]
            print(f"Detected: {object_name} ({confidence:.2f})")

    else:
        print("No RGB image data available.")

    # Process depth image
    depth_image = depth_camera.getRangeImage()
    if depth_image:
        depth_value = depth_image[0]  # Get depth at (0,0)
        print(f"Depth at [0,0]: {depth_value:.2f} meters")
    else:
        print("No depth data available.")

print("You can drive this robot using the keyboard arrows.")

initial_time = robot.getTime()

# Main loop
while robot.step(TIME_STEP) != -1:
    check_keyboard()
    # process_camera_data()
    objects = eurobin_camera.getRecognitionObjects()
    for obj in objects:
        print(f"Object ID: {obj.getId()}, Position: {obj.getPosition()}")

    # "Hello" movement
    time_elapsed = robot.getTime() - initial_time
    robot_parts["arm_6_joint"].setPosition(0.3 * math.sin(5.0 * time_elapsed) - 0.3)

# Cleanup OpenCV window when exiting
cv2.destroyAllWindows()


