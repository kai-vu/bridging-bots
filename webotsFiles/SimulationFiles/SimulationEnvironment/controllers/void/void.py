# void.py - Minimal Webots controller

# Import Webots' Supervisor module for compatibility
from controller import Supervisor

# Main function
def main():
    # Initialize the supervisor
    supervisor = Supervisor()
    
    # Get the simulation's basic time step
    time_step = int(supervisor.getBasicTimeStep())
    
    # Run the simulation loop
    while supervisor.step(time_step) != -1:
        # Do nothing (void controller)
        pass

# Entry point
if __name__ == "__main__":
    main()
S