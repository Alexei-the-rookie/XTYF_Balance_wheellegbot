# XTYF_Balance_wheellegbot
A visual pretest for XTYF's balance rifle in 2026.

# Working in Ubuntu 24.04 & ROS2 Jazzy environment.

# You should clone this repository to your local ROS2 workspace as follow:
git clone https://github.com/Alexei-the-rookie/XTYF_Balance_wheellegbot.git

# Then use these steps to open the Wheeleg-robot control sim:
colcon build --package-select balance_bot
source install/setup.bash
ros2 launch balance_bot control.launch.py

# Use it well for Robomaster development.
