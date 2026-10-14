import os

from ament_index_python.packages import (
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory(
        "acdcrp_segmentation"
    )

    default_parameters = os.path.join(
        package_share,
        "config",
        "live_segmentation.yaml",
    )

    parameters_file = LaunchConfiguration(
        "params_file"
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=default_parameters,
            description=(
                "Path to the live segmentation "
                "parameter YAML file"
            ),
        ),

        Node(
            package="acdcrp_segmentation",
            executable="inference_node",
            name="segmentation_inference_node",
            output="screen",
            parameters=[parameters_file],
        ),

        Node(
            package="acdcrp_segmentation",
            executable="evaluation_node",
            name="segmentation_evaluation_node",
            output="screen",
            parameters=[parameters_file],
        ),
    ])