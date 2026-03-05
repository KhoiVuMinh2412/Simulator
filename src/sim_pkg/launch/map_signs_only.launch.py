"""
Lightweight launch: Gazebo map + car + signs + traffic lights + ramp only.
Use this to quickly edit object positions without heavy models.
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import AnyLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    sim_pkg_dir = get_package_share_directory('sim_pkg')

    def _include(sublauncher: str):
        return IncludeLaunchDescription(
            AnyLaunchDescriptionSource(
                os.path.join(sim_pkg_dir, 'launch', 'sublaunchers', sublauncher)
            )
        )

    # Gazebo world (immediate)
    gazebo = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(sim_pkg_dir, 'launch', 'gazebo.launch')
        )
    )

    # Car (delay 5s)
    group1 = TimerAction(period=5.0, actions=[
        _include('car.launch'),
    ])

    # All signs (delay 10s)
    group2 = TimerAction(period=10.0, actions=[
        _include('enter_highway_signs.launch'),
        _include('leave_highway_signs.launch'),
        _include('prohibited_signs.launch'),
        _include('oneway_signs.launch'),
        _include('parking_signs.launch'),
        _include('priority_signs.launch'),
        _include('crosswalk_signs.launch'),
        _include('roundabout_signs.launch'),
        _include('stop_signs.launch'),
        _include('pedestrian_objects.launch'),
    ])

    # Traffic lights + ramp (delay 20s)
    group3 = TimerAction(period=20.0, actions=[
        _include('ramp.launch'),
        _include('traffic_lights.launch'),
    ])

    return LaunchDescription([
        gazebo,
        group1,
        group2,
        group3,
    ])
