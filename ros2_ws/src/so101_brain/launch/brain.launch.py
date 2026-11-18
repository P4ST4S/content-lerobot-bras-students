#!/usr/bin/env python3
"""brain.launch.py — launches the brain of the arm (IK + drawing).

    ros2 launch so101_brain brain.launch.py
    ros2 launch so101_brain brain.launch.py paper_z:=0.002 pen_length:=0.05
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    knobs = {"paper_z": "0.0", "pen_length": "0.05"}
    return LaunchDescription([
        *[DeclareLaunchArgument(k, default_value=v) for k, v in knobs.items()],
        Node(
            package="so101_brain",
            executable="brain",
            name="so101_brain",
            output="screen",
            parameters=[{
                "paper_z": ParameterValue(LaunchConfiguration("paper_z"), value_type=float),
                "pen_length": ParameterValue(LaunchConfiguration("pen_length"), value_type=float),
            }],
        ),
    ])
