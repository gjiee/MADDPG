import numpy as np
from parameters import *
import copy
import random
from cluster import *
from gym import spaces
import matplotlib.pyplot as plt

# 设置绘图的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class IoTDevice:
    '''
     generate_task：生成IoT设备上的不同任务
     compute_locally：本地计算时延
     offload_to_uav：将任务卸载到UAV的计算时延
    '''
    def __init__(self, device_id, position, task_type):
        """
        初始化 IoT 设备对象，包括设备 ID、位置和任务类型。
        """
        self.device_id = device_id
        self.position = np.array(position)  # (x, y)
        self.task_type = task_type  # 'low', 'medium', 'high'
        self.generate_task()
        self.connected_uav = None  # 连接的无人机
        self.cluster_label = None
        self.is_covered = False  # 初始化为未被覆盖
        self.is_task_completed = False  # 初始化任务未完成
        self.total_delay = 0.0  # 总时延
        self.is_task_processed = False  # 初始化任务未被处理

    def reset(self):
        """
        重置设备状态，用于环境初始化。
        """
        self.generate_task()
        self.connected_uav = None
        self.total_delay = 0
        self.cluster_label = None
        self.is_covered = False

    def copy(self):
        """
        创建设备对象的深度副本。
        """
        return copy.deepcopy(self)

    def generate_task(self):
        """
        根据任务类型生成任务的属性，包括数据大小、CPU 周期数、最大可容忍延迟和任务优先级。
        """
        if self.task_type == 'low':
            self.data_size = np.random.uniform(3e6, 4e6)
            self.cpu_cycles = np.random.uniform(400, 500)
            self.max_delay = np.random.uniform(5, 6)
            self.priority = 1
        elif self.task_type == 'medium':
            self.data_size = np.random.uniform(2e6, 3e6)
            self.cpu_cycles = np.random.uniform(600, 700)
            self.max_delay = np.random.uniform(2, 3)
            self.priority = 5
        else:  # 'high'
            self.data_size = np.random.uniform(1e6, 2e6)
            self.cpu_cycles = np.random.uniform(800, 1000)
            self.max_delay = np.random.uniform(1.0, 1.5)
            self.priority = 10
        self.local_compute_power = np.random.uniform(IOT_COMPUTE_POWER_MIN, IOT_COMPUTE_POWER_MAX)

    def generate_iot_devices(num_devices, seed=42):
        """
        生成指定数量的 IoT 设备，其中三个区域设备密集，其他区域较分散。

        参数：
        - num_devices: int, 设备数量

        返回：
        - devices: IoTDevice 对象的列表
        """
        np.random.seed(seed)
        random.seed(seed)
        devices = []

        # 定义高优先级和中等优先级任务的集中区域
        high_priority_areas = [
            (GROUND_LENGTH * 0.8, GROUND_WIDTH * 0.8),  # 区域1
            (GROUND_LENGTH * 0.2, GROUND_WIDTH * 0.2)  # 区域2
        ]
        medium_priority_area = (GROUND_LENGTH * 0.7, GROUND_WIDTH * 0.7)
        low_priority_area = (GROUND_LENGTH * 0.5, GROUND_WIDTH * 0.5)

        high_priority_task_count = int(num_devices * 0.3)

        for i in range(num_devices):
            task_type = np.random.choice(['low', 'medium', 'high'], p=[0.4, 0.3, 0.3])  # **调整任务类型比例**

            # 高优先级任务集中在指定区域
            if task_type == 'high':
                # 在两个高优先级区域均分任务
                if i < high_priority_task_count:
                    selected_area = high_priority_areas[0]  # 第一个区域
                else:
                    selected_area = high_priority_areas[1]  # 第二个区域

                x = np.random.normal(loc=selected_area[0], scale=GROUND_LENGTH * 0.1)
                y = np.random.normal(loc=selected_area[1], scale=GROUND_WIDTH * 0.05)
            # 中等优先级任务集中在另一个区域
            elif task_type == 'medium':
                x = np.random.normal(loc=medium_priority_area[0], scale=GROUND_LENGTH * 0.2)
                y = np.random.normal(loc=medium_priority_area[1], scale=GROUND_WIDTH * 0.1)
            # 低优先级任务分布在整个区域
            else:
                x = np.random.normal(loc=low_priority_area[0], scale=GROUND_LENGTH * 0.4)
                y = np.random.normal(loc=low_priority_area[1], scale=GROUND_WIDTH * 0.2)

            # 将生成的坐标限制在模拟区域范围内
            x = np.clip(x, 0, GROUND_LENGTH)
            y = np.clip(y, 0, GROUND_WIDTH)

            device = IoTDevice(device_id=i, position=(x, y), task_type=task_type)
            devices.append(device)

        return devices

    def compute_locally(self):
        """
        计算任务在本地执行所需的时间。
        """
        total_cycles = self.data_size * self.cpu_cycles
        local_computation_delay = total_cycles / self.local_compute_power
        self.total_delay = local_computation_delay

        # 标记任务已被处理
        self.is_task_processed = True

        # 判断任务是否在截止时间内完成
        self.is_task_completed = self.total_delay <= self.max_delay

        return self.total_delay

    def offload_to_uav(self, uav, compute_allocation_factor=1.0, bandwidth_allocation_factor=1.0):
        """
        将任务卸载到无人机执行，计算传输时间和无人机计算时间。
        - return: total_time: float，总的任务完成时间 = 传输时间 + 计算时间，单位：秒
        """
        if compute_allocation_factor <= 0 or bandwidth_allocation_factor <= 0:
            return float('inf')  # 无法分配资源，返回极大的时间

        # 计算传输速率
        transmission_rate = calculate_transmission_rate(
            self.position,
            uav.position,
            bandwidth_allocation_factor * UAV_BANDWIDTH,
        )
        transmission_time = self.data_size / max(transmission_rate, 1e-6)

        # 计算无人机计算时间
        uav_compute_power = uav.compute_power * compute_allocation_factor
        total_cycles = self.data_size * self.cpu_cycles
        uav_computation_time = total_cycles / max(uav_compute_power, 1e-6)

        # 总时延
        total_time = transmission_time + uav_computation_time
        self.total_delay = total_time

        # 标记任务已被处理
        self.is_task_processed = True

        # 判断任务是否在截止时间内完成
        self.is_task_completed = self.total_delay <= self.max_delay

        return self.total_delay

    def plot_iot_devices(devices, uavs=None, title=None):
        """
        绘制 IoT 设备的分布图，并可选绘制 UAV 的位置和覆盖范围。
        未被任何 UAV 覆盖的设备将以不同的颜色或形状表示。

        参数：
        - devices: IoTDevice 对象的列表
        - uavs: UAV 位置的列表，可选
        - title: 字符串，图表的标题，可选
        """
        plt.figure(figsize=(8, 8))
        colors = {'low': 'green', 'medium': 'orange', 'high': 'red'}
        labels = {'low': '低优先级任务', 'medium': '中优先级任务', 'high': '高优先级任务'}

        # 绘制被覆盖的设备
        for task_type in ['low', 'medium', 'high']:
            xs = [device.position[0] for device in devices if device.task_type == task_type and device.is_covered]
            ys = [device.position[1] for device in devices if device.task_type == task_type and device.is_covered]
            plt.scatter(xs, ys, c=colors[task_type], label=labels[task_type], alpha=0.6)

        # 绘制未被覆盖的设备
        xs_uncovered = [device.position[0] for device in devices if not device.is_covered]
        ys_uncovered = [device.position[1] for device in devices if not device.is_covered]
        if xs_uncovered:
            plt.scatter(xs_uncovered, ys_uncovered, c='gray', label='未覆盖设备', marker='x', alpha=0.6)

            # 绘制 UAV 位置和覆盖范围
        if uavs is not None and len(uavs) > 0:
            for idx, uav_position in enumerate(uavs):
                plt.scatter(uav_position[0], uav_position[1], c='blue', marker='^', s=100,
                            label='UAV' if idx == 0 else "")
                circle = plt.Circle((uav_position[0], uav_position[1]), UAV_COVER, color='blue', fill=False,
                                    linestyle='--',
                                    alpha=0.3)
                plt.gca().add_patch(circle)

        if title is not None:
            plt.title(title)
        else:
            plt.title('IoT 设备分布图及 UAV 覆盖范围')
        plt.xlabel('X 轴位置 (m)')
        plt.ylabel('Y 轴位置 (m)')
        plt.xlim(-100, GROUND_LENGTH + 100)
        plt.ylim(-100, GROUND_WIDTH + 100)
        plt.legend()
        plt.grid(True)
        plt.show()


class UAV:
    '''
     assign_device：将设备分配给UAV，绑定连接关系
     update_battery：更新UAV电池电量，当电量小于min值，逻辑有问题/////////部署的时候先不管无人机最小能量
     is_within_coverage：判断设备是否在UAV覆盖范围内
     compute_energy_consumption：计算无人机为连接的 IoT 设备执行任务时的总能耗
    '''
    def __init__(self, uav_id, position):
        """
        初始化 UAV 对象，包括 UAV ID、位置等参数。
        """
        self.uav_id = uav_id
        self.position = position
        self.height = UAV_HEIGHT
        self.coverage_radius = UAV_COVER
        self.compute_power = UAV_CPU_FREQUENCY
        self.bandwidth = UAV_BANDWIDTH
        self.connected_devices = []  # 已连接的设备列表
        self.battery_energy = UAV_BATTERY_ENERGY  # 电池能量
        self.min_energy = UAV_MIN_ENERGY  # 最小剩余能量
        self.speed = UAV_SPEED  # 飞行速度
        self.compute_allocation = {}  # 初始化计算资源分配策略
        self.bandwidth_allocation = {}  # 初始化带宽资源分配策略
        # 添加可用资源属性
        self.available_compute_resource = self.compute_power
        self.available_bandwidth = self.bandwidth

    def reset(self):
        """
        重置 UAV 状态，用于环境初始化。
        """
        # 不清空 connected_devices，保留分配关系
        self.battery_energy = UAV_BATTERY_ENERGY
        self.available_compute_resource = self.compute_power
        self.available_bandwidth = self.bandwidth
        self.compute_allocation = {}  # 重置计算资源分配策略
        self.bandwidth_allocation = {}  # 重置带宽资源分配策略

    def assign_device(self, device):
        """
        将设备分配给 UAV。
        """
        self.connected_devices.append(device)
        device.connected_uav = self
        device.is_covered = True   # 设置设备为已覆盖

    def is_within_coverage(self, device_position):
        """
        判断设备是否在 UAV 的覆盖范围内。
        """
        horizontal_distance = np.linalg.norm(device_position - self.position)
        return horizontal_distance <= self.coverage_radius

    def compute_energy_consumption(self, compute_allocation_factors=None):
        """
        计算 UAV 的能耗。

        参数：
        - compute_allocation_factors: dict，键为设备的 device_id，值为分配给该设备的计算资源比例（0到1之间）。
          如果没有提供分配比例，则默认按设备数量平均分配。

        返回：
        - total_energy: float，总能耗（焦耳）
        """
        total_energy = 0
        num_devices = len(self.connected_devices)
        if num_devices == 0:
            self.energy_consumed = total_energy
            return total_energy

        for device in self.connected_devices:
            # 获取分配比例
            if compute_allocation_factors and device.device_id in compute_allocation_factors:
                allocation_factor = compute_allocation_factors[device.device_id]
            else:
                allocation_factor = 1 / num_devices  # 平均分配

            # 计算能耗
            effective_compute_power = self.compute_power * allocation_factor
            total_cycles = device.data_size * device.cpu_cycles
            compute_time = total_cycles / max(effective_compute_power, 1e-6)
            energy = UAV_HARDWARE_CONSTANT * (effective_compute_power ** 2) * compute_time
            total_energy += energy

        self.energy_consumed = total_energy  # 更新能耗属性
        return total_energy

    def apply_action(self, action):
        """
        分配计算资源和带宽资源，确保资源分配一致性。
        """
        num_devices = len(self.connected_devices)
        if num_devices == 0:
            # print(f"UAV {self.uav_id}: No connected devices, skipping action.")
            return

        # 动作拆分
        compute_action = action[:MAX_DEVICES_RANDOM][:num_devices]
        bandwidth_action = action[MAX_DEVICES_RANDOM:2 * MAX_DEVICES_RANDOM][:num_devices]

        # 确保动作值非负
        compute_action = np.clip(compute_action, 0, None)
        bandwidth_action = np.clip(bandwidth_action, 0, None)

        # 归一化处理，确保总资源不超过 1.0
        total_compute_alloc = np.sum(compute_action)
        total_bandwidth_alloc = np.sum(bandwidth_action)

        if total_compute_alloc > 1.0:
            compute_action = compute_action / total_compute_alloc
        if total_bandwidth_alloc > 1.0:
            bandwidth_action = bandwidth_action / total_bandwidth_alloc

        # 根据设备优先级分配资源
        priority_indices = sorted(range(num_devices),
                                  key=lambda i: self.connected_devices[i].priority,
                                  reverse=True)

        compute_ratios = np.zeros(num_devices)
        bandwidth_ratios = np.zeros(num_devices)
        for rank, idx in enumerate(priority_indices):
            compute_ratios[idx] = compute_action[rank]
            bandwidth_ratios[idx] = bandwidth_action[rank]

        # 确保计算资源和带宽资源的同步分配
        for i in range(num_devices):
            if compute_ratios[i] < MIN_ALLOCATION_THRESHOLD or bandwidth_ratios[i] < MIN_ALLOCATION_THRESHOLD:
                compute_ratios[i] = 0.0
                bandwidth_ratios[i] = 0.0

        # 更新分配策略
        self.compute_allocation = {device.device_id: compute_ratios[i] for i, device in
                                   enumerate(self.connected_devices)}
        self.bandwidth_allocation = {device.device_id: bandwidth_ratios[i] for i, device in
                                     enumerate(self.connected_devices)}

        # 更新剩余资源
        self.available_compute_resource = self.compute_power * (1 - np.sum(compute_ratios))
        self.available_bandwidth = self.bandwidth * (1 - np.sum(bandwidth_ratios))

    def get_observation(self):
        """
        返回 UAV 的观测，包括自身状态和覆盖范围内设备的信息。
        """
        # 无人机自身状态
        uav_info = np.array([
            self.position[0],  # UAV 的 x 坐标
            self.position[1],  # UAV 的 y 坐标
            self.available_bandwidth,  # 可用带宽
            self.available_compute_resource,  # 可用计算资源
            self.battery_energy  # 电池剩余能量
        ])

        # 连接设备的信息
        devices_info = []
        for device in self.connected_devices[:MAX_DEVICES_RANDOM]:
            device_info = [
                device.position[0],  # 设备的 x 坐标
                device.position[1],  # 设备的 y 坐标
                device.data_size,  # 数据大小
                device.cpu_cycles,  # CPU 周期
                device.max_delay,  # 截止期限
                device.priority,  # 任务权重
            ]
            devices_info.extend(device_info)

        # 如果连接设备少于 MAX_DEVICES，使用零填充
        num_padded_devices = MAX_DEVICES_RANDOM - len(self.connected_devices)
        for _ in range(num_padded_devices):
            device_info = [0, 0, 0, 0, 0, 0]  # 包含位置坐标和任务信息
            devices_info.extend(device_info)

        devices_info = np.array(devices_info)

        # 合并观测
        observation = np.concatenate([uav_info, devices_info])

        return observation

class MultiUAVEnv:
    def __init__(self):
        """
            初始化 MultiUAV 环境，包括 IoT 设备和 UAV 的创建，以及设备与 UAV 的关联。
        """
        # 生成 IoT 设备列表
        self.devices = IoTDevice.generate_iot_devices(NUM_IOTS)

        # 执行 ISODATA 聚类，确定初始聚类数量和聚类中心
        isodata = ISODATA(self.devices)
        clusters_list, centroids = isodata.fit()

        # 执行 ISODATA_KMeans_Clustering，进一步优化 UAV 位置和设备分配
        clustering = ISODATA_Random_UAV_Deployment(
            devices=self.devices,
            initial_clusters=len(centroids),
        )
        device_assignments, uav_positions = clustering.fit()

        # 初始化 UAV 列表
        self.uavs = {}
        for idx, position in enumerate(uav_positions):
            uav_id = f'uav_{idx}'
            uav = UAV(uav_id=uav_id, position=position)
            self.uavs[uav_id] = uav

        # 将设备分配给对应的 UAV
        for uav_idx, devices_in_cluster in enumerate(device_assignments):
            uav_id = f'uav_{uav_idx}'
            uav = self.uavs[uav_id]
            for device in devices_in_cluster:
                uav.assign_device(device)

        # 绘制分布图
        uav_positions = [uav.position for uav in self.uavs.values()]
        IoTDevice.plot_iot_devices(self.devices, uavs=uav_positions, title="IoT设备与UAV分布")

        self.time_step = 0
        # 定义 action_spaces 和 observation_spaces
        self.action_spaces = {}
        self.observation_spaces = {}
        for uav_id, uav in self.uavs.items():
            # 定义每个 UAV 的观测空间
            obs_dim = 5 + MAX_DEVICES_RANDOM * 6  # 5 是 uav_info 的长度，6 是每个设备的信息长度（位置2 + 任务4）
            self.observation_spaces[uav_id] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32
            )

            # 定义每个 UAV 的动作空间
            action_dim = 2 * MAX_DEVICES_RANDOM  # 前 MAX_DEVICES 为计算资源分配，后 MAX_DEVICES 为带宽分配
            self.action_spaces[uav_id] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(action_dim,),
                dtype=np.float32
            )
        # 使用 spaces.Dict 定义 action_space 和 observation_space
        self.observation_space = spaces.Dict(self.observation_spaces)
        self.action_space = spaces.Dict(self.action_spaces)

    def reset(self):
        """
        重置环境状态，包括 UAV 和 IoT 设备的状态。
        """
        observations = {}
        for uav_id, uav in self.uavs.items():
            uav.reset()
            observations[uav_id] = uav.get_observation()

        for device in self.devices:
            device.generate_task()  # 只重置任务，不重置连接关系
            device.is_task_processed = False
            device.is_task_completed = False

        self.time_step = 0
        return observations

    def step(self, actions):
        """
        执行智能体的动作，更新环境状态，计算奖励。
        """
        # 为每个 UAV 分配资源
        for uav_id, action in actions.items():
            uav = self.uavs[uav_id]
            uav.apply_action(action)

        # 更新 IoT 设备状态
        for device in self.devices:
            if device.connected_uav is not None:
                uav = device.connected_uav
                # 获取资源分配比例
                compute_alloc = uav.compute_allocation.get(device.device_id, 0)
                bandwidth_alloc = uav.bandwidth_allocation.get(device.device_id, 0)

                # 本地计算时延
                local_delay = device.compute_locally()

                # 卸载计算时延（如果资源分配不足，返回极大值）
                if compute_alloc > 0 and bandwidth_alloc > 0:
                    offload_delay = device.offload_to_uav(uav, compute_alloc, bandwidth_alloc)
                else:
                    offload_delay = float('inf')

                # 优化选择：选择时延较小的方案
                if local_delay < offload_delay:
                    # 本地计算更优，回收 UAV 资源
                    device.total_delay = local_delay
                    device.is_task_completed = device.total_delay <= device.max_delay

                    # 回收无人机资源
                    if device.device_id in uav.compute_allocation:
                        uav.available_compute_resource += uav.compute_allocation.pop(device.device_id) * uav.compute_power
                    if device.device_id in uav.bandwidth_allocation:
                        uav.available_bandwidth += uav.bandwidth_allocation.pop(device.device_id) * uav.bandwidth
                else:
                    # 卸载到无人机
                    device.total_delay = offload_delay
                    device.is_task_completed = device.total_delay <= device.max_delay
            else:
                # 未连接无人机，本地计算
                device.total_delay = device.compute_locally()
                device.is_task_completed = device.total_delay <= device.max_delay

        # 计算所有 UAV 的能耗，找到最大能耗
        uav_energies = {}
        for uav_id, uav in self.uavs.items():
            eu = uav.compute_energy_consumption(
                compute_allocation_factors=uav.compute_allocation
            )
            uav_energies[uav_id] = eu
        max_eu = max(uav_energies.values()) if uav_energies else 0.0

        # 计算每个 UAV 的奖励
        rewards = {}
        for uav_id, uav in self.uavs.items():
            reward = self.compute_reward(uav, max_eu)
            rewards[uav_id] = reward

        # 获取新的观察
        observations = {}
        for uav_id, uav in self.uavs.items():
            observations[uav_id] = uav.get_observation()

        self.time_step += 1

        # 检查所有任务是否已被处理（无论成功或失败）
        all_tasks_processed = all([device.is_task_processed for device in self.devices])

        # 更新 dones
        dones = {uav_id: self.time_step >= MAX_TIME_STEPS or all_tasks_processed
                 for uav_id in self.uavs.keys()}
        # 添加 trunc 逻辑，基于任务超时限制
        truncs = {uav_id: self.time_step >= MAX_TIME_STEPS for uav_id in self.uavs.keys()}

        infos = {uav_id: {} for uav_id in self.uavs.keys()}

        return observations, rewards, dones, truncs, infos

    def compute_reward(self, uav, max_eu):
        """
        动态调整权重并优化时延惩罚。
        """
        w_energy = 0.001
        total_reward = 0

        # 计算所有连接设备的奖励
        for device in uav.connected_devices:
            # 时延满意度：延迟越小，奖励越高
            delay_satisfaction = (device.max_delay - device.total_delay) / device.max_delay

            # 将奖励加到总奖励上
            total_reward += delay_satisfaction

        # 计算 UAV 的能耗惩罚
        energy_penalty = w_energy * max_eu

        # 总奖励：时延奖励总和 - 能耗惩罚
        total_reward = total_reward - energy_penalty

        return total_reward

def calculate_transmission_rate(device_position, uav_position, bandwidth_per_user):
    """
    计算 IoT 设备与无人机之间的传输速率。

    参数：
    - device_position: 设备位置坐标（numpy 数组或 tuple）
    - uav_position: 无人机位置坐标（numpy 数组或 tuple）
    - bandwidth_per_user: 分配给每个用户的带宽（Hz）
    - UAV_HEIGHT: 无人机的飞行高度（米）
    - CARRIER_FREQUENCY: 载波频率（Hz）
    - SPEED_OF_LIGHT: 光速（米/秒）
    - IOT_TX_POWER: IoT 设备的发射功率（瓦特）
    - NOISE_POWER_DENSITY: 噪声功率密度（瓦特/Hz）

    返回值：
    - transmission_rate: 传输速率（bps）
    """
    # 计算 IoT 设备与无人机之间的距离和水平距离
    device_position = np.array(device_position)
    uav_position = np.array(uav_position)

    horizontal_distance = np.linalg.norm(device_position - uav_position)
    distance = np.sqrt(horizontal_distance ** 2 + UAV_HEIGHT ** 2)

    # 计算 IoT 设备与无人机之间的仰角
    if horizontal_distance == 0:
        horizontal_distance = 1e-6  # 防止除以零
    theta = np.arctan(UAV_HEIGHT / horizontal_distance)

    # 计算视距（LOS）传输的概率
    a = 9.61
    b = 0.16
    theta_deg = np.degrees(theta)  # 将弧度转换为度
    p_los = 1 / (1 + a * np.exp(-b * (theta_deg - a)))  # LOS 概率
    p_nlos = 1 - p_los  # 非视距（NLOS）概率

    # 自由空间路径损耗（FSPL）计算（单位：dB）
    fspl_db = 20 * np.log10(distance) + 20 * np.log10(CARRIER_FREQUENCY) + 20 * np.log10(4 * np.pi / SPEED_OF_LIGHT)
    fspl_linear = 10 ** (fspl_db / 10)

    # LOS 和 NLOS 的附加损耗因子（dB 转换为线性值）
    eta_los_db = 1.0  # LOS 情况，无附加损耗（0 dB）
    eta_nlos_db = 20.0  # NLOS 情况，增加 20 dB 的损耗
    eta_los = 10 ** (eta_los_db / 10)
    eta_nlos = 10 ** (eta_nlos_db / 10)

    # 总路径损耗（线性值，不是 dB）
    path_loss = p_los * fspl_linear * eta_los + p_nlos * fspl_linear * eta_nlos

    # 计算接收功率（线性值）
    received_power = IOT_TX_POWER / path_loss

    # 计算噪声功率
    noise_power = NOISE_POWER_DENSITY * bandwidth_per_user

    # 计算信噪比（SNR）
    snr = received_power / noise_power
    snr = max(snr, 1e-10)  # 防止 snr 为非正值

    # 计算传输速率，根据香农公式
    transmission_rate = bandwidth_per_user * np.log2(1 + snr)

    # 防止速率为负无穷或 NaN
    if np.isnan(transmission_rate) or transmission_rate <= 0:
        transmission_rate = 1e6  # 设置一个最低传输速率 1 Mbps

    return transmission_rate

