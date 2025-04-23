from controller import Robot, Camera, Motor
import os
import math

# User settings
rotation_steps = 8
rotation_angle_deg = 40
screenshot_dir = f"screenshots_{rotation_angle_deg}degrees_{rotation_steps}images"
wheel_radius = 0.033  # meters, typical for TIAGo LITE
axle_length = 0.16    # distance between wheels (meters)

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Find camera
camera = None
for i in range(robot.getNumberOfDevices()):
    dev = robot.getDeviceByIndex(i)
    if isinstance(dev, Camera):
        camera = dev
        break

if camera is None:
    raise RuntimeError("No camera found.")

camera.enable(timestep)

# Find wheel motors
left_motor = None
right_motor = None
for i in range(robot.getNumberOfDevices()):
    dev = robot.getDeviceByIndex(i)
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

# Screenshot folder setup
os.makedirs(screenshot_dir, exist_ok=True)

# Begin rotation and screenshotting
angle_rad_per_step = math.radians(rotation_angle_deg)

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
        robot.step(timestep)

    # Stop the motors
    left_motor.setVelocity(0)
    right_motor.setVelocity(0)

    # Wait for robot to stabilize
    for _ in range(10):
        robot.step(timestep)

    # Screenshot time
    filename = os.path.join(screenshot_dir, f"screenshot_{i:03d}.jpg")
    camera.saveImage(filename, 100)
    print(f"Saved: {filename}")
