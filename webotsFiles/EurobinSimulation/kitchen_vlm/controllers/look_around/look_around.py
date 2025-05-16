#!/home/user/pel_ws/pel_venv/bin/python

from controller import Robot, Camera, Motor, Supervisor
import os
import math
from rdflib import Graph, Namespace, Literal, RDF, URIRef
import datetime

ORKA = Namespace("http://w3id.org/def/orka#")
SOSA = Namespace("http://www.w3.org/ns/sosa/")
SSN = Namespace("http://www.w3.org/ns/ssn/")
OBOE = Namespace("http://ecoinformatics.org/oboe/oboe.1.0/oboe-core.owl#")
OBOE_CHAR = Namespace("http://ecoinformatics.org/oboe/oboe.1.2/oboe-characteristics.owl#")
def extract_color_from_node(node):
    shape = None
    children = node.getField("children")
    if not children:
        return None

    for i in range(children.getCount()):
        child = children.getMFNode(i)
        if child.getTypeName() == "Shape":
            shape = child
            break

    if not shape:
        return None

    appearance = shape.getField("appearance")
    if not appearance:
        return None

    appearance_node = appearance.getSFNode()
    if not appearance_node or appearance_node.getTypeName() != "Appearance":
        return None

    material = appearance_node.getField("material")
    if not material:
        return None

    material_node = material.getSFNode()
    if not material_node or material_node.getTypeName() != "Material":
        return None

    color_field = material_node.getField("diffuseColor")
    if color_field:
        return color_field.getSFColor()

    return None

def extract_shape_from_node(node):
    children = node.getField("children")
    if children:
        for i in range(children.getCount()):
            shape = children.getMFNode(i)
            if shape.getTypeName() == "Shape":
                geometry = shape.getField("geometry")
                if geometry:
                    return geometry.getSFNode().getTypeName()
    return None

def scan_world_objects(supervisor):
    root = supervisor.getRoot()
    children_field = root.getField("children")

    object_data = []
    for i in range(children_field.getCount()):
        node = children_field.getMFNode(i)
        if not node:
            continue

        node_id = node.getId()
        name_field = node.getField("name")
        if name_field:
            name = name_field.getSFString()
        else: name = ''
        typename = node.getTypeName()
        position = node.getPosition()
        color = extract_color_from_node(node)
        shape = extract_shape_from_node(node)

        object_data.append({
            "id": node_id,
            "name": name,
            "typename": typename,
            "position": [round(p, 3) for p in position] if position else None,
            "color": [round(c, 3) for c in color] if color else None,
            "shape": shape
        })

    return object_data

def create_obs_graph(owl_path, data, output_path="obs_graph.ttl"):
    g = Graph()
    g.parse(owl_path, format="xml")

    g.bind("orka", ORKA)
    g.bind("sosa", SOSA)
    g.bind("ssn", SSN)
    g.bind("oboe", OBOE)
    g.bind("oboe-char", OBOE_CHAR)


    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        

    robot_uri = URIRef(f"Tiago")
    g.add((robot_uri, RDF.type, ORKA.Robot))
    print(data)
    
    for obj in data:
        
        ent_uri = URIRef(f"{ORKA}Ent_{obj['id']}_{timestamp}")
        obs_uri = URIRef(f"{ORKA}Obs_{obj['id']}_{timestamp}")
        g.add((obs_uri, RDF.type, OBOE.Observation))
        g.add((ent_uri, RDF.type, OBOE.Entity))
        g.add((obs_uri, OBOE.ofEntity, ent_uri))
        g.add((obs_uri, ORKA.hasID, Literal(obj['id'])))
        
        if obj.get("name"):
            print(f"Adding name: {obj['name']}")
            char_name_uri = URIRef(f"{ORKA}char_name_{obj['id']}_{timestamp}")
            g.add((char_name_uri, RDF.type, OBOE_CHAR.Name))
            g.add((ent_uri, ORKA.hasCharacteristic, char_name_uri))
            g.add((char_name_uri, ORKA.hasValue, Literal(obj["name"])))
        if obj.get("typename"):
            print(f"Adding typename: {obj['typename']}")
            class_name_uri = URIRef(f"{ORKA}typename_{obj['id']}_{timestamp}")
            g.add((class_name_uri, RDF.type, ORKA.ObjectType))
            g.add((ent_uri, ORKA.hasCharacteristic, class_name_uri))
            g.add((class_name_uri, ORKA.hasValue, Literal(obj["typename"])))
        if obj.get("position"):
            print(f"Adding position: {obj['position']}")
            loc_uri = URIRef(f"{ORKA}Loc_{obj['id']}_{timestamp}")
            loc_str = Literal(",".join([str(round(v, 3)) for v in obj["position"]]))
            g.add((loc_uri, RDF.type, ORKA.Location))
            g.add((loc_uri, ORKA.hasValue, loc_str))
            g.add((ent_uri, ORKA.hasCharacteristic, loc_uri))
            g.add((ent_uri, ORKA.hasLocation, loc_uri))

    g.serialize(destination=output_path, format="turtle")
    print(f"Updated ontology written to {output_path}")



# User settings
rotation_steps = 8
rotation_angle_deg = 40
# fov = 1.047 # typical camera
# fov = 2.097 # for humans
fov = 1.4
screenshot_dir = f"screenshots"
wheel_radius = 0.033  # meters, typical for TIAGo LITE
axle_length = 0.16    # distance between wheels (meters)


# Screenshot folder setup
os.makedirs(screenshot_dir, exist_ok=True)

# Begin rotation and screenshotting
angle_rad_per_step = math.radians(rotation_angle_deg)

supervisor = Supervisor()
TIME_STEP = int(supervisor.getBasicTimeStep())

print("Scanning world...")
supervisor.step(TIME_STEP)
data = scan_world_objects(supervisor)

create_obs_graph('orka.owl', data)


timestep = int(supervisor.getBasicTimeStep())

# Find camera
camera = None
for i in range(supervisor.getNumberOfDevices()):
    dev = supervisor.getDeviceByIndex(i)
    if isinstance(dev, Camera):
        camera = dev
        break

if camera is None:
    raise RuntimeError("No camera found.")

camera.enable(timestep)
camera.setFov(fov)
# Find wheel motors
left_motor = None
right_motor = None
for i in range(supervisor.getNumberOfDevices()):
    dev = supervisor.getDeviceByIndex(i)
    if isinstance(dev, Motor):
        name = dev.getName()
        if ("left" in name) and ("wheel" in name):
            left_motor = dev
        elif ("right" in name) and ("wheel" in name):
            right_motor = dev

if left_motor is None or right_motor is None:
    raise RuntimeError("Could not find both wheel motors. Do you even turn, bro?")

# Set motors to velocity control mode
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))


for i in range(rotation_steps):
    # Compute angular velocity to rotate in place
    # v = ω * r ⇒ ω = v / r
    rotation_speed = 0.5  # radians/sec robot angular speed
    wheel_speed = (rotation_speed * axle_length / 2) / wheel_radius  # convert to wheel speed

    # Set wheel velocities: opposite directions
    left_motor.setVelocity(wheel_speed)
    right_motor.setVelocity(-wheel_speed)

    # Duration needed to rotate desired angle: t = θ / ω
    duration = angle_rad_per_step / rotation_speed
    steps = int(duration * 1000 / timestep)

    for _ in range(steps):
        supervisor.step(timestep)

    # Stop the motors
    left_motor.setVelocity(0)
    right_motor.setVelocity(0)

    # Wait for robot to stabilize
    for _ in range(10):
        supervisor.step(timestep)

    # Screenshot time
    filename = os.path.join(screenshot_dir, f"screenshot_{i:03d}.jpg")
    camera.saveImage(filename, 100)
    print(f"Saved: {filename}")
