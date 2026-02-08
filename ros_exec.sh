#!/bin/bash
#
# This script runs ROS 2 commands in a clean environment
# to avoid conflicts with the Poetry Python venv.

# Unset environment variables set by Poetry/direnv
unset VIRTUAL_ENV
unset POETRY_ACTIVE
unset PYTHONPATH

# Force the PATH to prioritize system binaries and ROS
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/ros/humble/bin"

# Source the required ROS setup files
source /opt/ros/humble/setup.bash
source ./install/local_setup.bash > /dev/null 2>&1 || true

# --- Debugging ---
echo "--- Inside ros_exec.sh ---"
echo "Using python: $(which python3)"
echo "Python version: $(python3 --version)"
echo "--------------------------"

# Execute the command passed to this script
exec "$@"
