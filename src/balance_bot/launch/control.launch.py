# nano lqr_control.launch.py
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    pkg_share = FindPackageShare('balance_bot').find('balance_bot')

    # 使用带传感器的URDF
    urdf_file = os.path.join(pkg_share, 'urdf', 'bot_with_sensors.xacro')

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                get_package_share_directory('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ]),
        launch_arguments={'gz_args': '-r empty.sdf'}.items()
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'use_sim_time': True,
            'robot_description': ParameterValue(Command(['xacro ', urdf_file]), value_type=str)
        }]
    )

    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        parameters=[{'use_sim_time': True}]
    )

    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'wheel_leg_robot',
            '-topic', 'robot_description',
            '-x', '0.0', '-y', '0.0', '-z', '0.5'
        ]
    )

    controller_manager = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster']
    )

    controller_spawners = []
    controllers = [
        'left_hip_joint_controller',
        'left_knee_joint_controller',
        'left_wheel_joint_controller',
        'right_hip_joint_controller',
        'right_knee_joint_controller',
        'right_wheel_joint_controller'
    ]

    for controller in controllers:
        controller_spawners.append(Node(
            package='controller_manager',
            executable='spawner',
            arguments=[controller]
        ))

    # Bridge Gazebo IMU to ROS
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/imu@sensor_msgs/msg/Imu@gz.msgs.IMU'],
        output='screen'
    )

    # 使用纯LQR控制器
    lqr_controller = Node(
        package='balance_bot',
        executable='initial_pose_controller.py',
        output='screen'
    )

    # 键盘控制（可选）
    teleop_twist_keyboard = Node(
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        name='teleop_twist_keyboard',
        output='screen',
        prefix='xterm -e'
    )

    return LaunchDescription([
        gazebo_launch,
        robot_state_publisher,
        joint_state_publisher,
        spawn_entity,
        controller_manager,
        *controller_spawners,
        bridge,
        lqr_controller,
        teleop_twist_keyboard,
    ])