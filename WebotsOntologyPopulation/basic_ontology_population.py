#!/usr/bin/env python
from controller import Robot, Camera, Lidar, Motor
import math
import rdflib
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD
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

# Webots robot initialization
robot = Robot()

# --- Cameras ---
rgb_camera = robot.getDevice("Astra rgb")
rgb_camera.enable(TIME_STEP)

depth_camera = robot.getDevice("Astra depth")
depth_camera.enable(TIME_STEP)

eurobin_camera = robot.getDevice("camera")
eurobin_camera.enable(TIME_STEP)
eurobin_camera.recognitionEnable(TIME_STEP)

# --- Lidar ---
lidar = robot.getDevice("Hokuyo URG-04LX-UG01")
lidar.enable(TIME_STEP)
lidar.enablePointCloud()

# Allow sensor stabilization
for _ in range(10):
    robot.step(TIME_STEP)

# --- Motors ---
left_motor = robot.getDevice("wheel_left_joint")
right_motor = robot.getDevice("wheel_right_joint")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# ------------------------------
# ONTOLOGY SETUP
# ------------------------------
ontology_file = "C:/Users/francesca/Documents/eurobin_ontology.ttl"
g = Graph()
g.parse(ontology_file, format="turtle")

# Use your defined namespace
EX = Namespace("http://example.org/kitchen#")

# Dictionary: recognized label (first word) → existing ontology class name
# (Classes must exist in your ontology with exactly these names, e.g., :Appliance, :Furniture, etc.)
label_to_class = {
    "fridge":         "Appliance",
    "oven":           "Appliance",
    "table":          "Furniture",
    "sink":           "Furniture",
    "dishwasher":     "Appliance",
    "trash_bin":      "Container",
    "plate":          "Utensil",
    "kitchen_towel":  "KitchenAccessory"
}

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def sanitize_label(raw_label):
    """
    Convert something like 'kitchen towel' → 'kitchen_towel'
    """
    return raw_label.lower().strip().replace(" ", "_")

def first_word(label_str):
    """
    Return the first word of label_str (lowercased).
    E.g., "fridge door" -> "fridge"
    """
    return label_str.lower().split()[0]

def check_obstacles():
    data = lidar.getRangeImage()
    if not data:
        return False
    center = len(data) // 2
    slice_data = data[center - 10 : center + 10]
    return (np.mean(slice_data) < SAFE_DISTANCE)

def find_individual_by_label(label_str):
    """
    SPARQL query to return an individual's URI if ex:hasName equals label_str.
    """
    query = f"""
    PREFIX ex: <{EX}>
    SELECT ?obj
    WHERE {{
      ?obj ex:hasName "{label_str}" .
    }}
    """
    results = g.query(query)
    for row in results:
        return row.obj
    return None

def get_or_create_individual(label_str):
    """
    If an individual with ex:hasName = label_str exists, return it.
    Otherwise, if the first word is in label_to_class, create a new individual
    (named after the sanitized label) typed as the mapped class.
    Skip unknown labels.
    """
    existing = find_individual_by_label(label_str)
    if existing:
        return existing

    norm_label = sanitize_label(label_str)
    fw = norm_label.split("_")[0]
    if fw not in label_to_class:
        print(f"Label '{label_str}' not in dictionary; skipping creation.")
        return None

    mapped_class = label_to_class[fw]  # e.g., "Appliance"
    # Check that this class exists in the ontology:
    if (EX[mapped_class], RDF.type, OWL.Class) not in g:
        print(f"Mapped class '{mapped_class}' not found in ontology; skipping '{label_str}'.")
        return None

    # Create new individual with name exactly as the sanitized label.
    individual_uri = EX[norm_label]
    g.add((individual_uri, RDF.type, EX[mapped_class]))
    g.add((individual_uri, EX.hasName, Literal(label_str)))
    g.add((individual_uri, EX.hasColor, Literal("unknown")))
    g.add((individual_uri, EX.hasDimensions, Literal("unknown")))
    print(f"Created new individual '{norm_label}' as {mapped_class} with label '{label_str}'.")
    return individual_uri

def update_individual_location(obj_uri, pos):
    """
    Update the individual's location using proper data property assertions.
    Now uses EX.hasCoordinate_X and EX.hasCoordinate_Y (with underscores)
    instead of annotation properties.
    """
    loc_uri = EX[f"{obj_uri.split('#')[-1]}_Location"]
    # Remove old location data
    g.remove((loc_uri, None, None))
    g.remove((obj_uri, EX.hasLocation, None))
    g.add((loc_uri, RDF.type, EX.Location))
    # Use data properties as proper assertions
    g.add((loc_uri, EX.hasCoordinate_X, Literal(pos[0], datatype=XSD.float)))
    g.add((loc_uri, EX.hasCoordinate_Y, Literal(pos[2], datatype=XSD.float)))
    g.add((obj_uri, EX.hasLocation, loc_uri))
    print(f"Updated location for {obj_uri} to ({pos[0]:.2f}, {pos[2]:.2f}).")

def update_individual_orientation(obj_uri, orientation_value):
    """
    Update the individual's orientation using the data property hasCoordinate_Z.
    Create (or update) an Orientation individual and link it via hasOrientation.
    """
    orient_uri = EX[f"{obj_uri.split('#')[-1]}_Orientation"]
    g.remove((orient_uri, None, None))
    g.add((orient_uri, RDF.type, EX.Orientation))
    g.add((orient_uri, EX.hasCoordinate_Z, Literal(orientation_value, datatype=XSD.float)))
    g.remove((obj_uri, EX.hasOrientation, None))
    g.add((obj_uri, EX.hasOrientation, orient_uri))
    print(f"Updated orientation for {obj_uri} to {orientation_value:.2f}.")

def process_recognition(obj):
    """
    Process each recognized object:
    - Get or create an individual for that label.
    - Update its location and orientation.
    """
    raw_label = obj.getModel()  # e.g., "fridge" or "fridge door"
    pos = obj.getPosition()     # [x, y, z]
    # For orientation, we assume pos[1] (or use a different method if available)
    orientation_value = pos[1]

    # For this version, we skip special door handling; process everything as normal.
    ind_uri = get_or_create_individual(raw_label)
    if ind_uri:
        update_individual_location(ind_uri, pos)
        update_individual_orientation(ind_uri, orientation_value)

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
                process_recognition(obj)

    step_counter += 1

left_motor.setVelocity(0)
right_motor.setVelocity(0)
print("Simulation ended.")

# Save final ontology
g.serialize(destination="C:/Users/francesca/Documents/kitchen_ontology_updated.ttl", format="turtle")
print("Updated ontology saved as 'kitchen_ontology_updated.ttl'.")



