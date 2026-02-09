#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import math
import numpy as np
from scipy.linalg import solve_continuous_are

class PureLQRController(Node):
    def __init__(self):
        super().__init__('pure_lqr_controller')

        # 状态变量 (10维状态向量)
        self.theta_bl = 0.0                # 左侧腿角
        self.dtheta_bl = 0.0            # 左侧腿角变化率
        self.theta_br = 0.0              # 右侧腿角
        self.dtheta_br = 0.0          # 右侧腿角变化率
        self.theta_wl = 0.0          # 左腿与轮夹角
        self.dtheta_wl = 0.0      # 左腿与轮夹角变化率
        self.theta_wr = 0.0          # 右腿与轮夹角
        self.dtheta_wr = 0.0      # 右腿角速度
        self.theta_b = 0.0           # 机体倾斜角 [rad]
        self.dtheta_b = 0.0       # 机体倾斜角速度
        self.roll = 0.0           # 机体翻滚角

        # 膝关节状态
        self.theta_kl = 0.0      # 左膝关节角
        self.dtheta_kl = 0.0     # 左膝关节角速度
        self.theta_kr = 0.0      # 右膝关节角
        self.dtheta_kr = 0.0     # 右膝关节角速度

        # 传感器观测数据 (力矩)
        self.tau_hip_l = 0.0     # 左髋关节观测力矩
        self.tau_hip_r = 0.0     # 右髋关节观测力矩
        self.tau_knee_l = 0.0    # 左膝关节观测力矩
        self.tau_knee_r = 0.0    # 右膝关节观测力矩
        self.tau_wheel_l = 0.0   # 左轮观测力矩
        self.tau_wheel_r = 0.0   # 右轮观测力矩

        # 传感器观测数据 (IMU加速度)
        self.acc_x = 0.0
        self.acc_y = 0.0
        self.acc_z = 0.0

        # 参考状态
        self.theta_bl_ref = 0.0                # 左侧腿角
        self.dtheta_bl_ref = 0.0            # 左侧腿角变化率
        self.theta_br_ref = 0.0              # 右侧腿角
        self.dtheta_br_ref = 0.0          # 右侧腿角变化率
        self.theta_wl_ref = 0.0          # 左腿与轮夹角
        self.dtheta_wl_ref = 0.0      # 左腿与轮夹角变化率
        self.theta_wr_ref = 0.0          # 右腿与轮夹角
        self.dtheta_wr_ref = 0.0      # 右腿角速度
        self.theta_b_ref = 0.0           # 机体倾斜角 [rad]
        self.dtheta_b_ref = 0.0       # 机体倾斜角速度

        # LQR控制输入 (6维控制向量)
        self.T_lw_l = 0.0    # 左轮力矩 [Nm]
        self.T_lw_r = 0.0    # 右轮力矩 [
        self.T_bl_l = 0.0    # 左腿髋关节力矩 [Nm]
        self.T_bl_r = 0.0    # 右腿髋关节力矩 [Nm]

        # LQR参数
        self.Q = None  # 状态权重矩阵 (8x8)
        self.R = None  # 控制权重矩阵 (6x6)
        self.K = None  # LQR增益矩阵 (6x8)

        # 系统物理参数 (System parameters updated)
        self.m_b = 15.0               # 机体质量 [kg]
        self.g = 9.81                 # 重力加速度 [m/s²]
        self.R_w = 0.058              # 轮子半径 [m]
        self.I_b = 0.1125             # 机体对质心俯仰转动惯量 [kg·m²]
        self.I_b_yaw = 0.05           # 机体偏航转动惯量 [kg·m²]
        self.I_w = 0.001              # 轮子转动惯量 [kg·m²]
        self.I_l = 0.018              # 腿转动惯量 [kg·m²]

        # 腿部参数
        self.l_thigh = 0.230          # 大腿长 [m]
        self.l_shank = 0.287          # 小腿长 [m]
        self.l_com_to_hip = 0.087     # 腿部质心距机体(髋) [m]
        self.l_com_to_wheel = 0.172   # 腿部质心距轮轴 [m]

        # 虚拟腿相关
        # 髋关节距轮子轴高度 0.176 (用于目标设定)
        self.target_height = 0.176
        self.half_wheel_track = 0.175 # 二分之一轮距 [m]

        self.m_w = 0.982              # 轮子质量 [kg]
        self.m_l = 2.047              # 腿部质量 [kg]

        # Initialize joint variables
        self.theta_bl_joint_pos = 0.0
        self.dtheta_bl_joint_vel = 0.0
        self.theta_kl = 0.0
        self.dtheta_kl = 0.0

        self.theta_br_joint_pos = 0.0
        self.dtheta_br_joint_vel = 0.0
        self.theta_kr = 0.0
        self.dtheta_kr = 0.0

        # 控制限制
        self.max_wheel_velocity = 10.0   # 最大轮子速度 [rad/s]
        self.max_torque = 15.0           # 最大关节力矩 [Nm]

        # 初始化LQR
        self.init_lqr()

        # 创建发布器
        self.left_hip_pub = self.create_publisher(Float64MultiArray, '/left_hip_joint_controller/commands', 10)
        self.left_knee_pub = self.create_publisher(Float64MultiArray, '/left_knee_joint_controller/commands', 10)
        self.left_wheel_pub = self.create_publisher(Float64MultiArray, '/left_wheel_joint_controller/commands', 10)

        self.right_hip_pub = self.create_publisher(Float64MultiArray, '/right_hip_joint_controller/commands', 10)
        self.right_knee_pub = self.create_publisher(Float64MultiArray, '/right_knee_joint_controller/commands', 10)
        self.right_wheel_pub = self.create_publisher(Float64MultiArray, '/right_wheel_joint_controller/commands', 10)

        # 创建订阅器
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10)

        #self.odom_sub = self.create_subscription(
        #    Odometry,
        #    '/wheel_leg_robot/odometry',
        #    self.odom_callback,
        #    10)

        # 关节状态订阅器
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
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

    def cmd_vel_callback(self, msg):
        """处理速度控制命令"""
        # 暂时没用到，留空或实现基本逻辑
        pass

    def joint_state_callback(self, msg):
        """处理关节状态数据"""
        try:
            # 建立映射以应对不同的joint顺序
            name_map = {name: i for i, name in enumerate(msg.name)}

            # 左腿
            if 'left_hip_joint' in name_map:
                idx = name_map['left_hip_joint']
                self.theta_bl_joint_pos = msg.position[idx]
                self.dtheta_bl_joint_vel = msg.velocity[idx]
                self.tau_hip_l = msg.effort[idx]

            if 'left_knee_joint' in name_map:
                idx = name_map['left_knee_joint']
                self.theta_kl = msg.position[idx]
                self.dtheta_kl = msg.velocity[idx]
                self.tau_knee_l = msg.effort[idx]

            if 'left_wheel_joint' in name_map:
                idx = name_map['left_wheel_joint']
                self.theta_wl = msg.position[idx]
                self.dtheta_wl = msg.velocity[idx]
                self.tau_wheel_l = msg.effort[idx]

            # 右腿
            if 'right_hip_joint' in name_map:
                idx = name_map['right_hip_joint']
                self.theta_br_joint_pos = msg.position[idx]
                self.dtheta_br_joint_vel = msg.velocity[idx]
                self.tau_hip_r = msg.effort[idx]

            if 'right_knee_joint' in name_map:
                idx = name_map['right_knee_joint']
                self.theta_kr = msg.position[idx]
                self.dtheta_kr = msg.velocity[idx]
                self.tau_knee_r = msg.effort[idx]

            if 'right_wheel_joint' in name_map:
                idx = name_map['right_wheel_joint']
                self.theta_wr = msg.position[idx]
                self.dtheta_wr = msg.velocity[idx]
                self.tau_wheel_r = msg.effort[idx]

        except Exception as e:
            self.get_logger().error(f"Joint State Parse Error: {e}")

    def init_lqr(self):
        """初始化LQR控制器参数"""
        # 状态向量 x (10维):
        # [0] theta_bl      (左虚拟腿角)
        # [1] dtheta_bl     (左虚拟腿角速度)
        # [2] theta_br      (右虚拟腿角)
        # [3] dtheta_br     (右虚拟腿角速度)
        # [4] theta_wl      (左轮角度)
        # [5] dtheta_wl     (左轮角速度)
        # [6] theta_wr      (右轮角度)
        # [7] dtheta_wr     (右轮角速度)
        # [8] theta_b       (机体俯仰角)
        # [9] dtheta_b      (机体俯仰角速度)

        # 控制向量 u (4维):
        # [0] T_lw_l        (左轮力矩)
        # [1] T_lw_r        (右轮力矩)
        # [2] T_bl_l        (左腿虚拟转动力矩)
        # [3] T_bl_r        (右腿虚拟转动力矩)

        # 1. 状态权重矩阵 Q (10x10)
        q_diag = [
            10.0, 1.0,    # 左腿虚拟角
            10.0, 1.0,    # 右腿虚拟角
            1.0,  0.5,    # 左轮
            1.0,  0.5,    # 右轮
            100.0, 10.0   # 机体俯仰 (平衡最重要)
        ]
        self.Q = np.diag(q_diag)

        # 2. 控制权重矩阵 R (4x4)
        self.R = np.diag([
            5.0, 5.0,     # 轮子力矩
            3.0, 3.0      # 虚拟腿转动力矩
        ])

        # 3. 构建系统矩阵 A (10x10)
        A = self.build_system_matrix()

        # 4. 构建控制矩阵 B (10x4)
        B = self.build_control_matrix()

        # 5. 求解连续时间代数Riccati方程
        try:
            P = solve_continuous_are(A, B, self.Q, self.R)
            self.K = np.linalg.inv(self.R) @ B.T @ P
            self.get_logger().info("LQR增益矩阵计算成功")
            self.get_logger().info(f"K矩阵维度: {self.K.shape} (应为 4x10)")
        except Exception as e:
            self.get_logger().error(f"LQR增益计算失败: {e}")
            self.K = None

    def build_system_matrix(self):
        """构建系统矩阵 A (10x10)"""
        # 简化模型：双轮倒立摆 + 腿部摆动
        A = np.zeros((10, 10))

        # 运动学关系: angle_dot = angle_velocity
        A[0, 1] = 1.0  # d(theta_bl) = dtheta_bl
        A[2, 3] = 1.0  # d(theta_br) = dtheta_br
        A[4, 5] = 1.0  # d(theta_wl) = dtheta_wl
        A[6, 7] = 1.0  # d(theta_wr) = dtheta_wr
        A[8, 9] = 1.0  # d(theta_b)  = dtheta_b

        # 动力学近似
        # 这是一个高度简化的线性化模型，用于LQR求解
        # 实际上应该基于更精确的动力学方程

        # 机体俯仰动力学 (倒立摆)
        # d(dtheta_b) ~ g/L * theta_b
        gravity_term = self.g / self.target_height # 使用目标高度作为倒立摆长度估计
        A[9, 8] = gravity_term

        # 腿部摆动动力学 (复摆)
        # d(dtheta_leg) ~ -g/L_leg * theta_leg
        # 使用腿部质心距离
        leg_swing_term = -self.g / self.l_com_to_hip
        A[1, 0] = leg_swing_term
        A[3, 2] = leg_swing_term

        return A

    def build_control_matrix(self):
        """构建控制矩阵 B (10x4)"""
        # 控制输入: [T_lw_l, T_lw_r, T_bl_l, T_bl_r]
        # 状态: [th_bl, dth_bl, th_br, dth_br, th_wl, dth_wl, th_wr, dth_wr, th_b, dth_b]

        B = np.zeros((10, 4))

        # 惯量参数近似
        inv_I_w = 1.0 / self.I_w       # 轮子
        inv_I_b = 1.0 / self.I_b       # 机体
        inv_I_l = 1.0 / self.I_l       # 腿部

        # 1. 轮子力矩 T_lw (Left at col 0, Right at col 1)
        # 对轮子加速
        B[5, 0] = inv_I_w
        B[7, 1] = inv_I_w
        # 对机体产生反作用力 (导致俯仰)
        B[9, 0] = -inv_I_b
        B[9, 1] = -inv_I_b

        # 2. 腿部虚拟力矩 T_bl (Left at col 2, Right at col 3)
        # 对腿部加速 (注意方向定义)
        # T_bl 是髋关节施加在虚拟杆上的力矩
        B[1, 2] = inv_I_l
        B[3, 3] = inv_I_l
        # 对机体产生反作用力
        B[9, 2] = -inv_I_b
        B[9, 3] = -inv_I_b

        return B

    def imu_callback(self, msg):
        """处理IMU数据"""
        # 读取线加速度
        self.acc_x = msg.linear_acceleration.x
        self.acc_y = msg.linear_acceleration.y
        self.acc_z = msg.linear_acceleration.z

        # 从四元数提取欧拉角
        x = msg.orientation.x
        y = msg.orientation.y
        z = msg.orientation.z
        w = msg.orientation.w

        # 四元数到欧拉角转换
        # 翻滚角 (roll)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        self.roll = math.atan2(sinr_cosp, cosr_cosp)

        # 俯仰角 (pitch)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            self.theta_b = math.copysign(math.pi / 2, sinp)
        else:
            self.theta_b = math.asin(sinp)

        # 偏航角 (yaw)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

        # 角速度
        self.roll_vel = msg.angular_velocity.x
        self.pitch_vel = msg.angular_velocity.y
        self.yaw_vel = msg.angular_velocity.z

    def get_state_vector(self):
        """获取当前状态向量 (10维)"""
        # 计算虚拟腿部状态
        self.update_virtual_leg_states()

        return np.array([
            self.theta_bl,      # [0] 左虚拟腿角
            self.dtheta_bl,     # [1] 左虚拟腿角速度
            self.theta_br,      # [2] 右虚拟腿角
            self.dtheta_br,     # [3] 右虚拟腿角速度
            self.theta_wl,      # [4] 左轮角度
            self.dtheta_wl,     # [5] 左轮角速度
            self.theta_wr,      # [6] 右轮角度
            self.dtheta_wr,     # [7] 右轮角速度
            self.theta_b,       # [8] 机体俯仰角 (theta_b)
            self.dtheta_b       # [9] 机体俯仰角速度
        ])

    def update_virtual_leg_states(self):
        """计算虚拟腿状态 (正运动学)"""
        # 几何参数
        l1 = 0.23   # 大腿
        l2 = 0.287  # 小腿

        # 左腿
        q1_l = self.theta_bl_joint_pos # 需要在joint_callback中读取真实髋关节角
        q2_l = self.theta_kl           # 真实膝关节角

        # 虚拟腿向量 (相对于髋关节)
        # x = -l1*sin(q1) - l2*sin(q1+q2)
        # z = -l1*cos(q1) - l2*cos(q1+q2)
        x_l = -l1 * math.sin(q1_l) - l2 * math.sin(q1_l + q2_l)
        z_l = -l1 * math.cos(q1_l) - l2 * math.cos(q1_l + q2_l)

        self.L_virtual_l = math.sqrt(x_l**2 + z_l**2)
        self.theta_bl = math.atan2(-x_l, -z_l) # 虚拟腿角度

        # 雅可比计算用于速度
        J_l = self.compute_virtual_jacobian(q1_l, q2_l)
        dq_l = np.array([self.dtheta_bl_joint_vel, self.dtheta_kl])
        v_virtual_l = J_l @ dq_l # [dL, dtheta]
        self.DL_virtual_l = v_virtual_l[0]
        self.dtheta_bl = v_virtual_l[1]

        # 右腿
        q1_r = self.theta_br_joint_pos
        q2_r = self.theta_kr

        x_r = -l1 * math.sin(q1_r) - l2 * math.sin(q1_r + q2_r)
        z_r = -l1 * math.cos(q1_r) - l2 * math.cos(q1_r + q2_r)

        self.L_virtual_r = math.sqrt(x_r**2 + z_r**2)
        self.theta_br = math.atan2(-x_r, -z_r)

        J_r = self.compute_virtual_jacobian(q1_r, q2_r)
        dq_r = np.array([self.dtheta_br_joint_vel, self.dtheta_kr])
        v_virtual_r = J_r @ dq_r
        self.DL_virtual_r = v_virtual_r[0]
        self.dtheta_br = v_virtual_r[1]

    def compute_virtual_jacobian(self, q1, q2):
        """计算从关节空间到虚拟腿空间的雅可比矩阵"""
        l1 = 0.23
        l2 = 0.287

        s1 = math.sin(q1)
        c1 = math.cos(q1)
        s12 = math.sin(q1 + q2)
        c12 = math.cos(q1 + q2)

        x = -l1 * s1 - l2 * s12
        z = -l1 * c1 - l2 * c12
        L2 = x**2 + z**2
        L = math.sqrt(L2)

        # Partial derivatives of x, z w.r.t q1, q2
        dxdq1 = -l1 * c1 - l2 * c12
        dxdq2 = -l2 * c12
        dzdq1 = l1 * s1 + l2 * s12
        dzdq2 = l2 * s12

        # L = sqrt(x^2 + z^2)
        # dL/dq = (x*dx/dq + z*dz/dq) / L
        dLdq1 = (x * dxdq1 + z * dzdq1) / L
        dLdq2 = (x * dxdq2 + z * dzdq2) / L

        # theta = atan2(-x, -z)
        # dtheta/dq = (-(dz/dq)*(-x) - (-dx/dq)*(-z)) / L^2
        #           = (x*dz/dq - z*dx/dq) / L^2
        dthdq1 = (x * dzdq1 - z * dxdq1) / L2
        dthdq2 = (x * dzdq2 - z * dxdq2) / L2

        return np.array([
            [dLdq1, dLdq2],
            [dthdq1, dthdq2]
        ])

    def lqr_control(self, state):
        """纯LQR控制计算 + 虚拟力转换"""
        if self.K is not None:
            # u = -K * x (4x1)
            u = -self.K @ state

            T_wheel_l = u[0]
            T_wheel_r = u[1]
            T_virtual_rot_l = u[2]
            T_virtual_rot_r = u[3]

            # --- 高度控制 (独立于LQR) ---
            # 施加沿虚拟腿方向的力 F
            # 目标高度 (根据题目要求)
            L_ref = self.target_height

            # 高度PD控制 + 重力补偿
            kp_h = 1000.0
            kd_h = 50.0
            F_gravity = self.m_b * self.g / 2.0 / math.cos(self.theta_bl) # 简单分配

            F_l = kp_h * (L_ref - self.L_virtual_l) + kd_h * (0 - self.DL_virtual_l) + F_gravity
            F_r = kp_h * (L_ref - self.L_virtual_r) + kd_h * (0 - self.DL_virtual_r) + F_gravity

            # --- 雅可比转置映射 ---
            # tau = J^T * [F, T_rot]^T

            # 左腿
            J_l = self.compute_virtual_jacobian(self.theta_bl_joint_pos, self.theta_kl)
            traj_forces_l = np.array([F_l, T_virtual_rot_l])
            joint_torques_l = J_l.T @ traj_forces_l

            # 右腿
            J_r = self.compute_virtual_jacobian(self.theta_br_joint_pos, self.theta_kr)
            traj_forces_r = np.array([F_r, T_virtual_rot_r])
            joint_torques_r = J_r.T @ traj_forces_r

            # 赋值
            self.cmd_wheel_torque_l = T_wheel_l
            self.cmd_wheel_torque_r = T_wheel_r

            self.cmd_hip_torque_l = joint_torques_l[0]
            self.cmd_knee_torque_l = joint_torques_l[1]

            self.cmd_hip_torque_r = joint_torques_r[0]
            self.cmd_knee_torque_r = joint_torques_r[1]

            # 限幅
            self.limit_torques()

        else:
            self.fallback_pd_control(state)

    def limit_torques(self):
        self.cmd_wheel_torque_l = np.clip(self.cmd_wheel_torque_l, -self.max_torque, self.max_torque)
        self.cmd_wheel_torque_r = np.clip(self.cmd_wheel_torque_r, -self.max_torque, self.max_torque)
        self.cmd_hip_torque_l   = np.clip(self.cmd_hip_torque_l,   -self.max_torque, self.max_torque)
        self.cmd_hip_torque_r   = np.clip(self.cmd_hip_torque_r,   -self.max_torque, self.max_torque)
        self.cmd_knee_torque_l  = np.clip(self.cmd_knee_torque_l,  -self.max_torque, self.max_torque)
        self.cmd_knee_torque_r  = np.clip(self.cmd_knee_torque_r,  -self.max_torque, self.max_torque)


    def fallback_pd_control(self, state):
        """备用PD控制器 (当LQR不可用时)"""
        # 简单解耦PD

        # 1. 俯仰平衡 (主要靠轮子力矩)
        pitch_err = state[4]
        pitch_vel = state[5]
        T_bal = -(15.0 * pitch_err + 3.0 * pitch_vel)

        # 2. 偏航控制 (轮子差速)
        yaw_err = state[6]
        yaw_vel = state[7]
        T_yaw = -(5.0 * yaw_err + 1.0 * yaw_vel)

        # 3. 高度控制 (靠腿部推力) - 极其简化
        h_err = state[2]
        h_vel = state[3]
        F_h = -(100.0 * h_err + 10.0 * h_vel)
        # 简单的力分配: 假设力均匀分配给髋和膝 (实际需要雅可比)
        T_leg = F_h * 0.1

        self.cmd_wheel_torque_l = T_bal - T_yaw
        self.cmd_wheel_torque_r = T_bal + T_yaw

        self.cmd_hip_torque_l = T_leg
        self.cmd_hip_torque_r = T_leg
        self.cmd_knee_torque_l = T_leg
        self.cmd_knee_torque_r = T_leg

    def control_loop(self):
        """主控制循环"""
        # 获取当前状态
        state = self.get_state_vector()

        # 计算控制量 (力矩)
        self.lqr_control(state)

        # 发布控制命令 (全部为力矩)
        self.publish_joint_command(self.left_wheel_pub,  self.cmd_wheel_torque_l)
        self.publish_joint_command(self.right_wheel_pub, self.cmd_wheel_torque_r)

        self.publish_joint_command(self.left_hip_pub,    self.cmd_hip_torque_l)
        self.publish_joint_command(self.right_hip_pub,   self.cmd_hip_torque_r)

        self.publish_joint_command(self.left_knee_pub,   self.cmd_knee_torque_l)
        self.publish_joint_command(self.right_knee_pub,  self.cmd_knee_torque_r)

        # 调试信息
        if int(self.get_clock().now().nanoseconds / 1e9) % 2 == 0: # 提高日志频率
            self.get_logger().info(
                f"Pitch: {math.degrees(self.theta_b):.1f}deg | "
                f"Torques(Nm) -> W_L:{self.cmd_wheel_torque_l:.2f} W_R:{self.cmd_wheel_torque_r:.2f} "
                f"H_L:{self.cmd_hip_torque_l:.2f} K_L:{self.cmd_knee_torque_l:.2f}"
            )

    def publish_joint_command(self, publisher, value):
        msg = Float64MultiArray()
        msg.data = [float(value)]
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