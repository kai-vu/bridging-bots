import subprocess
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

obs_dir = os.path.join(current_dir, "observation-graph")
obs_script = "main.py"

action_dir = os.path.join(current_dir, "action-graph")
action_script = "main.py"

subprocess.run(["python", obs_script], cwd=obs_dir)
subprocess.run(["python", action_script], cwd=action_dir)