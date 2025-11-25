#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import math
import numpy as np
from scipy.linalg import solve_continuous_are

class PureLQRController(Node):
    def __init__(self):
        super().__init__('pure_lqr_controller')

        # 状态变量 (6维状态向量)
        self.theta = 0.0          # 机体俯仰角 [rad]
        self.theta_dot = 0.0      # 机体俯仰角速度 [rad/s]
        self.x_pos = 0.0          # 前后方向位置 [m]
        self.x_vel = 0.0          # 前后方向速度 [m/s]
        self.phi = 0.0            # 腿与竖直方向夹角 [rad]
        self.phi_dot = 0.0        # 腿与竖直方向夹角速度 [rad/s]

        # 参考状态
        self.x_ref = 0.0          # 参考位置 [m]
        self.x_vel_ref = 0.0      # 参考速度 [m/s]
        self.psi_dot_ref = 0.0    # 参考偏航角速度 [rad/s]
        self.gamma_ref =0.0       # 参考横滚姿态角 [rad]

        # LQR控制输入 (2维控制向量)
        self.T = 0.0              # 轮子的转矩 [Nm]
        self.T_p =0.0             # 关节转矩 [Nm]

        # LQR参数
        self.Q = None  # 状态权重矩阵 (8x8)
        self.R = None  # 控制权重矩阵 (6x6)
        self.K = None  # LQR增益矩阵 (6x8)

        # 系统物理参数
        self.mass = 3.0              # 机体质量 [kg]
        self.g = 9.81                # 重力加速度 [m/s²]
        self.wheel_radius = 0.08     # 轮子半径 [m]
        self.I_body = 0.1            # 机体转动惯量 [kg·m²]
        self.I_wheel = 0.01          # 轮子转动惯量 [kg·m²]
        self.I_leg = 0.02            # 腿转动惯量 [kg·m²]
        self.leg_length = 0.45       # 腿长度(经过计算得到) [m]
        self.L = 0.2                 # 腿重心到轮子的长度 [m]
        self.L_m = 0.3               # 腿重心到髋关节的长度 [m]
        self.mass_w = 0.01           # 轮子重量 [kg]
        self.mass_leg = 0.5          # 单腿重量 [kg]

        # 控制限制
        self.max_wheel_velocity = 10.0   # 最大轮子速度 [rad/s]
        self.max_torque = 15.0           # 最大关节力矩 [Nm]

        # 初始化LQR
        self.init_lqr()

        # 创建发布器
        self.left_hip_pub = self.create_publisher(Float64, '/left_hip_joint/command', 10)
        self.left_knee_pub = self.create_publisher(Float64, '/left_knee_joint/command', 10)
        self.left_wheel_pub = self.create_publisher(Float64, '/left_wheel_joint/command', 10)

        self.right_hip_pub = self.create_publisher(Float64, '/right_hip_joint/command', 10)
        self.right_knee_pub = self.create_publisher(Float64, '/right_knee_joint/command', 10)
        self.right_wheel_pub = self.create_publisher(Float64, '/right_wheel_joint/command', 10)

        # 创建订阅器
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10)

        self.odom_sub = self.create_subscription(
            Odometry,
            '/wheel_leg_robot/odometry',
            self.odom_callback,
            10)

        # 创建速度命令订阅器
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10)

        # 控制定时器
        self.control_timer = self.create_timer(0.02, self.control_loop)  # 50Hz控制频率

        self.get_logger().info("纯LQR轮腿机器人控制器已启动")

    def init_lqr(self):
        """初始化LQR控制器参数"""
        # 状态向量: [theta, theta_dot, x_pos, x_vel, phi, phi_dot]
        # 控制向量: [T, T_p]

        # 状态权重矩阵 Q (6x6)
        # 对角元素分别对应各个状态的权重
        self.Q = np.diag([
            80.0,   # theta - 平衡最重要
            25.0,   # theta_dot - 俯仰角速度控制
            0.5,    # x_pos - 位置控制权重较低
            2.0,    # x_vel - 速度控制中等权重
            8.0,    # phi - 腿角控制
            3.0     # phi_dot - 腿角速度控制
        ])

        # 控制权重矩阵 R (2x2)
        self.R = np.diag([
            0.02,   # T - 轮子力矩控制权重
            0.02,   # T_p - 关节力矩控制权重
        ])

        # 构建系统矩阵 A (6x6)
        # 这是简化的线性化模型，实际应用中需要通过系统辨识得到
        A = self.build_system_matrix()

        # 构建控制矩阵 B (6x2)
        B = self.build_control_matrix()

        # 求解连续时间代数Riccati方程
        try:
            P = solve_continuous_are(A, B, self.Q, self.R)
            self.K = np.linalg.inv(self.R) @ B.T @ P
            self.get_logger().info("LQR增益矩阵计算成功")
            self.get_logger().info(f"K矩阵形状: {self.K.shape}")
        except Exception as e:
            self.get_logger().error(f"LQR增益计算失败: {e}")
            # 使用经验PD控制器作为备用
            self.K = None

    def build_system_matrix(self):
        """构建系统矩阵 A"""
        # 简化的线性化模型，在平衡点附近
        #[theta, theta_dot, x_pos, x_vel, phi, phi_dot]
        A = np.zeros((6, 6))

        # 位置和速度关系
        A[0, 1] = 1    # theta_dot = theta_dot
        A[1, 0] = 1    # A_1 计算
        A[1, 4] = 1    # A_2 计算
        A[2, 3] = 1    # x_dot = x_vel
        A[3, 0] = 1    # A_3 计算
        A[3, 4] = 1    # A_4 计算
        A[4, 5] = 1    # phi_dot = phi_dot
        A[5, 0] = 1    # A_5 计算
        A[5, 4] = 1    # A_6 计算

        return A

    def build_control_matrix(self):
        """构建控制矩阵 B"""
        B = np.zeros((8, 6))

        # 控制输入对状态的影响
        B[1, 0] = 1    # B_1 计算
        B[1, 1] = 1    # B_2 计算
        B[3, 0] = 1    # B_3 计算
        B[3, 1] = 1    # B_4 计算
        B[5, 0] = 1    # B_5 计算
        B[5, 1] = 1    # B_6 计算
        return B

    def imu_callback(self, msg):
        """处理IMU数据"""
        # 从四元数提取欧拉角
        x = msg.orientation.x
        y = msg.orientation.y
        z = msg.orientation.z
        w = msg.orientation.w

        # 四元数到欧拉角转换
        # 俯仰角 (pitch)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            self.pitch = math.copysign(math.pi / 2, sinp)
        else:
            self.pitch = math.asin(sinp)

        # 偏航角 (yaw)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

        # 角速度
        self.roll_vel = msg.angular_velocity.x
        self.pitch_vel = msg.angular_velocity.y
        self.yaw_vel = msg.angular_velocity.z

    def odom_callback(self, msg):
        """处理里程计数据"""
        self.x_pos = msg.pose.pose.position.x
        self.x_vel = msg.twist.twist.linear.x

    def cmd_vel_callback(self, msg):
        """处理速度命令"""
        self.x_vel_ref = msg.linear.x

        # 根据角速度命令调整参考偏航角
        if abs(msg.angular.z) > 0.01:
            self.yaw_ref += msg.angular.z * 0.02

    def get_state_vector(self):
        """获取当前状态向量"""
        return np.array([
            self.x_pos - self.x_ref,
            self.x_vel - self.x_vel_ref,
            self.height - self.height_ref,
            self.height_vel,
            self.pitch - self.pitch_ref,
            self.pitch_vel,
            self.yaw - self.yaw_ref,
            self.yaw_vel
        ])

    def lqr_control(self, state):
        """纯LQR控制计算"""
        if self.K is not None:
            # u = -K * x
            control = -self.K @ state

            # 分离控制量
            self.left_wheel_velocity = control[0]
            self.right_wheel_velocity = control[1]
            self.left_hip_torque = control[2]
            self.right_hip_torque = control[3]
            self.left_knee_torque = control[4]
            self.right_knee_torque = control[5]

            # 控制量限幅
            self.left_wheel_velocity = np.clip(self.left_wheel_velocity,
                                               -self.max_wheel_velocity,
                                               self.max_wheel_velocity)
            self.right_wheel_velocity = np.clip(self.right_wheel_velocity,
                                                -self.max_wheel_velocity,
                                                self.max_wheel_velocity)

            self.left_hip_torque = np.clip(self.left_hip_torque,
                                           -self.max_torque,
                                           self.max_torque)
            self.right_hip_torque = np.clip(self.right_hip_torque,
                                            -self.max_torque,
                                            self.max_torque)
            self.left_knee_torque = np.clip(self.left_knee_torque,
                                            -self.max_torque,
                                            self.max_torque)
            self.right_knee_torque = np.clip(self.right_knee_torque,
                                             -self.max_torque,
                                             self.max_torque)
        else:
            # 备用PD控制器
            self.fallback_pd_control(state)

    def fallback_pd_control(self, state):
        """备用PD控制器 (当LQR不可用时)"""
        # 平衡控制 (俯仰)
        kp_pitch = 60.0
        kd_pitch = 12.0
        pitch_torque = -(kp_pitch * state[4] + kd_pitch * state[5])

        # 高度控制
        kp_height = 120.0
        kd_height = 25.0
        height_torque = -(kp_height * state[2] + kd_height * state[3])

        # 速度控制
        kp_vel = 8.0
        wheel_velocity = kp_vel * state[1]

        # 偏航控制
        kp_yaw = 15.0
        kd_yaw = 3.0
        yaw_velocity = -(kp_yaw * state[6] + kd_yaw * state[7])

        # 分配控制量
        self.left_wheel_velocity = wheel_velocity - yaw_velocity
        self.right_wheel_velocity = wheel_velocity + yaw_velocity
        self.left_hip_torque = pitch_torque + height_torque
        self.right_hip_torque = pitch_torque + height_torque
        self.left_knee_torque = -height_torque
        self.right_knee_torque = -height_torque

    def control_loop(self):
        """主控制循环"""
        # 获取当前状态
        state = self.get_state_vector()

        # LQR控制计算
        self.lqr_control(state)

        # 发布控制命令
        # 轮子使用速度控制
        self.publish_joint_command(self.left_wheel_pub, self.left_wheel_velocity)
        self.publish_joint_command(self.right_wheel_pub, self.right_wheel_velocity)

        # 关节使用力矩控制
        self.publish_joint_command(self.left_hip_pub, self.left_hip_torque)
        self.publish_joint_command(self.right_hip_pub, self.right_hip_torque)
        self.publish_joint_command(self.left_knee_pub, self.left_knee_torque)
        self.publish_joint_command(self.right_knee_pub, self.right_knee_torque)

        # 调试信息
        if int(self.get_clock().now().nanoseconds / 1e9) % 5 == 0:
            self.get_logger().info(
                f"状态: pos={self.x_pos:.3f}, vel={self.x_vel:.3f}, "
                f"height={self.height:.3f}, pitch={math.degrees(self.pitch):.2f}°, "
                f"yaw={math.degrees(self.yaw):.2f}°"
            )
            self.get_logger().info(
                f"控制: L_wheel={self.left_wheel_velocity:.2f}, R_wheel={self.right_wheel_velocity:.2f}, "
                f"L_hip={self.left_hip_torque:.2f}, R_hip={self.right_hip_torque:.2f}"
            )

    def publish_joint_command(self, publisher, value):
        msg = Float64()
        msg.data = float(value)
        publisher.publish(msg)

def main():
    rclpy.init()
    controller = PureLQRController()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()