import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'acdcrp_segmentation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
        (
            os.path.join(
                "share",
                package_name,
                "launch",
            ),
            glob("launch/*.launch.py"),
        ),
        (
            os.path.join(
                "share",
                package_name,
                "config",
            ),
            glob("config/*.yaml"),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lufei Wang',
    maintainer_email='lufei.wang@rwth-aachen.de',
    description='ROS 2 live semantic segmentation inference and evaluation for CARLAB.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        "console_scripts": [
            "inference_node = "
            "acdcrp_segmentation.inference_node:main",
            "evaluation_node = "
            "acdcrp_segmentation.evaluation_node:main",
        ],
    },
)
