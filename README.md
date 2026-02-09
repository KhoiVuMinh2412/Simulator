# Autonomous Driving Simulator

This repository contains a ROS 2 based simulation environment for autonomous driving development. It includes a Gazebo world, a car model with physics plugins, various track objects (traffic lights, signs, pedestrians), and example nodes for control and lane detection.

## Project Structure

```text
├── src/
│   ├── example/            # Example nodes for camera processing and car control
│   ├── models_pkg/         # SDF models for the car, track, and environment objects
│   ├── plugins_pkgs/       # Gazebo plugins (C++) for car physics, GPS, IMU, etc.
│   ├── sim_pkg/            # Launch files and world definitions
│   └── traffic_light_pkg/  # Logic for traffic light control
├── justfile                # Command runner shortcuts (recommended)
├── requirements.txt        # Python dependencies
└── ros_exec.sh             # Wrapper script to run commands in the ROS 2 environment
```

## Prerequisites

*   **OS**: Linux (Ubuntu 22.04 recommended)
*   **ROS 2 Distribution**: [Humble Hawksbill](https://docs.ros.org/en/humble/Installation.html)
*   **Tools**:
    *   `python3` (3.10+)
    *   `colcon` (Build tool)
    *   `just` (Command runner - Optional but recommended)

## Installation

1.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Install Just** (Optional, for easier command execution):
    ```bash
    # See https://github.com/casey/just for official instructions
    sudo apt install just
    ```

3.  **Build the Workspace**:
    Make sure you have sourced your ROS 2 installation (e.g., `source /opt/ros/humble/setup.bash`).
    ```bash
    # Using the provided wrapper (handles sourcing automatically)
    ./ros_exec.sh colcon build
    ```
    **Notice:** **Don't use** the manual ros2 build and source command as the python version will conflict:
    
    ~~colcon build~~\
    ~~source install/setup.bash~~
 
    #### **All commands must be run with the ./ros_exec.sh at the front!!!**

    *Tip: If you only modified one package, you can build just that one:*
    ```bash
    ./ros_exec.sh colcon build --packages-select <package_name>
    ```

## Running the Simulation

This project uses `just` to simplify running complex ROS launch commands. If you don't have `just`, you can look at the `justfile` to see the underlying `ros2 launch` commands.

### 1. Launch the Simulation
You can launch the map with just the car, or with all environment objects (signs, pedestrians, etc.).

*   **Car Only**:
    ```bash
    just car-only
    ```
    *Equivalent to: `ros2 launch sim_pkg map_with_car.launch`*

*   **All Objects**:
    ```bash
    just all-objects
    ```
    *Equivalent to launching `map_with_car.launch`, waiting 5s, and then launching `all_objects.launch`.*

### 2. Run Example Nodes
Once the simulation is running, you can run the example nodes in a separate terminal.

*   **Lane Keeping / Control**:
    Runs the lane detection and pure pursuit controller.
    ```bash
    just control_example
    ```

*   **Camera Visualization**:
    Runs a node that processes the camera feed.
    ```bash
    just camera_example
    ```

## Development Notes

*   **`ros_exec.sh`**: This script is a helper wrapper that ensures commands run with the correct ROS 2 environment variables and Python path, avoiding conflicts with system Python environments or virtualenvs (like poetry).
*   **Permissions**: If you cannot run scripts, ensure they are executable:
    ```bash
    chmod +x ros_exec.sh
    ```
