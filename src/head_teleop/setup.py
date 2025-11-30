from setuptools import setup
from glob import glob

package_name = 'head_teleop'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Team',
    maintainer_email='your@email.com',
    description='Head-controlled UR7e with blink commands',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'head_pose_blink = head_teleop.head_pose_blink_node:main',
            'head_mapper = head_teleop.head_teleop_mapper_node:main',
        ],
    },
)