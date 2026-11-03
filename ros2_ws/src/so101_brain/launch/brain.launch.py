#!/usr/bin/env python3
"""brain.launch.py — launches the brain of the arm (IK + pick & place + drawing).

    ros2 launch so101_brain brain.launch.py                   # pick & place autostart
    ros2 launch so101_brain brain.launch.py autostart:=false  # drawing mode
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    knobs = {"autostart": "true", "paper_z": "0.0", "pen_length": "0.03"}
    return LaunchDescription([
        *[DeclareLaunchArgument(k, default_value=v) for k, v in knobs.items()],
        Node(
            package="so101_brain",
            executable="brain",
            name="so101_brain",
            output="screen",
            parameters=[{
                "autostart": ParameterValue(LaunchConfiguration("autostart"), value_type=bool),
                "paper_z": ParameterValue(LaunchConfiguration("paper_z"), value_type=float),
                "pen_length": ParameterValue(LaunchConfiguration("pen_length"), value_type=float),
            }],
        ),
    ])
