#!/usr/bin/env python
from controller import Robot, Camera, Lidar, Motor
from WebotsOntologyPopulation import BasicOntologyPopulator
import numpy as np

# ------------------------------
# CONFIGURATION
# ------------------------------
TIME_STEP = 8
SIMULATION_DURATION = 30.0
RECOGNITION_INTERVAL = 100
SAFE_DISTANCE = 1.0
TURN_SPEED = 2.0
FORWARD_SPEED = 3.0

# File paths and ontology setup
ontology_file = "C:/Users/francesca/Documents/eurobin_ontology.ttl"
updated_ontology_file = "C:/Users/francesca/Documents/eurobin_ontology_updated.ttl"
namespace_uri = "http://example.org/kitchen#"

label_to_class = {
    "fridge":         "Appliance",
    "oven":           "Appliance",
    "table":          "Furniture",
    "sink":           "Furniture",
    "dishwasher":     "Appliance",
    "trash bin":      "Container",
    "plate":          "Utensil",
    "kitchen towel":  "KitchenAccessory",
    "bowl":           "Utensil",
    "glass":          "Utensil",
    "milk bottle":    "FoodItem",
    "dishwasher door":"Component",
    "fridge door":    "Component",
    "freezer door":   "Component",
    "chair":          "Furniture",
    "hot plate":      "Appliance"
}

# ------------------------------
# ROBOT INITIALIZATION
# ------------------------------
robot = Robot()

rgb_camera = robot.getDevice("Astra rgb")
rgb_camera.enable(TIME_STEP)

depth_camera = robot.getDevice("Astra depth")
depth_camera.enable(TIME_STEP)

eurobin_camera = robot.getDevice("camera")
eurobin_camera.enable(TIME_STEP)
eurobin_camera.recognitionEnable(TIME_STEP)

lidar = robot.getDevice("Hokuyo URG-04LX-UG01")
lidar.enable(TIME_STEP)
lidar.enablePointCloud()

for _ in range(10):
    robot.step(TIME_STEP)

left_motor = robot.getDevice("wheel_left_joint")
right_motor = robot.getDevice("wheel_right_joint")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# ------------------------------
# ONTOLOGY HANDLER INIT
# ------------------------------
ontology_handler = BasicOntologyPopulator(ontology_file, namespace_uri, label_to_class)

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def check_obstacles():
    data = lidar.getRangeImage()
    if not data:
        return False
    center = len(data) // 2
    slice_data = data[center - 10 : center + 10]
    return (np.mean(slice_data) < SAFE_DISTANCE)

# ------------------------------
# MAIN LOOP
# ------------------------------
step_counter = 0
start_time = robot.getTime()

while robot.step(TIME_STEP) != -1:
    current_time = robot.getTime()
    if current_time - start_time >= SIMULATION_DURATION:
        break

    if check_obstacles():
        left_motor.setVelocity(-TURN_SPEED)
        right_motor.setVelocity(TURN_SPEED)
    else:
        left_motor.setVelocity(FORWARD_SPEED)
        right_motor.setVelocity(FORWARD_SPEED)

    if step_counter % RECOGNITION_INTERVAL == 0:
        recognized_list = eurobin_camera.getRecognitionObjects()
        if recognized_list:
            for obj in recognized_list:
                ontology_handler.process_recognition(obj)

    step_counter += 1

left_motor.setVelocity(0)
right_motor.setVelocity(0)
print("Simulation ended.")

# Save final ontology
ontology_handler.save_ontology(updated_ontology_file)
print("Updated ontology saved as 'eurobin_ontology_updated.ttl'.")


