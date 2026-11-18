"""robot_state_publisher for the SO-ARM101.

    ros2 launch so101_description viz.launch.py            # waits for /joint_states
    ros2 launch so101_description viz.launch.py demo:=true # joints frozen at 0 (smoke test)

Publishes /robot_description and the TFs of the arm. Visualization: run rviz2
from the control container (X11) — see docker/docker-compose.yml.

IMPORTANT: without a node publishing /joint_states (your driver_node!), the joint
TFs do not exist → the arm appears "in pieces" in RViz. This is expected,
not a bug: publish sensor_msgs/JointState from your driver.
demo:=true launches a fake joint_state_publisher (zero pose) to check
the display chain independently of your code.
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    urdf = Path(get_package_share_directory("so101_description")) / "urdf" / "so101.urdf"
    robot_description = urdf.read_text()

    return LaunchDescription([
        DeclareLaunchArgument("demo", default_value="false",
                              description="Publish fake joint_states (zero pose)"),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
        ),
        # world → base_link: same as the MuJoCo frame (robot base at the world origin)
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=["--frame-id", "world", "--child-frame-id", "base_link"],
        ),
        # Smoke test only — in normal use, /joint_states comes from the driver_node.
        Node(
            package="joint_state_publisher",
            executable="joint_state_publisher",
            condition=IfCondition(LaunchConfiguration("demo")),
        ),
    ])
