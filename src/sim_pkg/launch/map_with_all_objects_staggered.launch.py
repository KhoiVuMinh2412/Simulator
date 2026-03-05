"""
Staggered launch for Gazebo simulation with all objects.

This Python launch file is equivalent to map_with_all_objects.launch but
staggers the spawn_entity calls in groups with delays so that Gazebo's
/spawn_entity service has time to become available and process requests
without being overwhelmed.
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import (
    AnyLaunchDescriptionSource,
)
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

    # ── Group 0: Gazebo world (immediate) ──────────────────────────────
    gazebo = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(sim_pkg_dir, 'launch', 'gazebo.launch')
        )
    )

    # ── Group 1: Car (delay 5s — wait for Gazebo to be ready) ─────────
    group1 = TimerAction(period=5.0, actions=[
        _include('car.launch'),
    ])

    # ── Group 2: Road signs batch 1 (delay 10s) — 11 entities ───────
    group2 = TimerAction(period=10.0, actions=[
        _include('enter_highway_signs.launch'),
        _include('leave_highway_signs.launch'),
        _include('prohibited_signs.launch'),
        _include('oneway_signs.launch'),
    ])

    # ── Group 3: Parking + priority signs (delay 18s) — 8 entities ──
    group3 = TimerAction(period=18.0, actions=[
        _include('parking_signs.launch'),
        _include('priority_signs.launch'),
    ])

    # ── Group 4: Crosswalk + roundabout signs (delay 25s) — 9 entities
    group4 = TimerAction(period=25.0, actions=[
        _include('crosswalk_signs.launch'),
        _include('roundabout_signs.launch'),
    ])

    # ── Group 5: Stop signs + pedestrian objects (delay 32s) ────────
    group5 = TimerAction(period=32.0, actions=[
        _include('stop_signs.launch'),
        _include('pedestrian_objects.launch'),
    ])

    # ── Group 6: (delay 40s) ─────────────────────────────────────────
    # obstacle_car and pedestrian_objects removed to reduce Gazebo load

    # ── Group 7: Members removed to reduce Gazebo load ──

    # ── Group 8: Traffic lights + misc (delay 60s) — 7 entities ─────
    group8 = TimerAction(period=60.0, actions=[
        _include('ramp.launch'),
        # roadblock removed to reduce Gazebo load
        _include('traffic_lights.launch'),
    ])

    return LaunchDescription([
        gazebo,
        group1,
        group2,
        group3,
        group4,
        group5,
        group8,
    ])
