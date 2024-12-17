# env.py

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
    def __init__(self, device_id, position, task_type):
        self.device_id = device_id
        self.position = np.array(position)  # 设备的二维位置 (x, y)
        self.task_type = task_type
        self.connected_uav = None
        self.is_covered = False  # 表示是否与 UAV 关联
        self.is_within_coverage = False  # 新增：表示是否在覆盖范围内
        self.is_task_processed = False
        self.is_task_completed = False
        self.offload_fraction = 0.0  # 初始卸载比例
        self.total_delay = 0.0  # 总延迟
        self.snr = 0.0  # 初始 SNR
        self.generate_task()  # 在初始化时生成一次任务

    def reset(self):
        """
        重置设备状态，用于环境初始化。
        """
        self.generate_task()
        self.connected_uav = None
        self.total_delay = 0.0
        self.is_covered = False
        self.is_task_processed = False
        self.is_task_completed = False
        self.offload_fraction = 0.0

    def copy(self):
        return copy.deepcopy(self)

    def generate_task(self):
        """
        根据任务类型生成任务的属性，包括数据大小、CPU 周期数、最大可容忍延迟和任务优先级。
        """
        if self.task_type == 'low':
            self.data_size = np.random.uniform(5e6, 6e6)
            self.cpu_cycles = np.random.uniform(600, 800)
            self.max_delay = np.random.uniform(2, 3)
            self.priority = 1
        elif self.task_type == 'medium':
            self.data_size = np.random.uniform(2e6, 3e6)
            self.cpu_cycles = np.random.uniform(1000, 1200)
            self.max_delay = np.random.uniform(1, 2)
            self.priority = 5
        else:  # 'high'
            self.data_size = np.random.uniform(2e6, 3e6)
            self.cpu_cycles = np.random.uniform(1000, 1200)
            self.max_delay = np.random.uniform(1, 2)
            self.priority = 5
        self.local_compute_power = np.random.uniform(IOT_COMPUTE_POWER_MIN, IOT_COMPUTE_POWER_MAX)
        self.offload_fraction = 0.0  # 重置卸载比例
        self.total_delay = 0.0  # 重置总延迟
        self.is_task_processed = False
        self.is_task_completed = False

    @staticmethod
    def generate_iot_devices(num_devices, seed=42):
        """
        生成指定数量的 IoT 设备，其中三个区域设备密集，其他区域较分散。
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
            task_type = np.random.choice(['low', 'medium', 'high'], p=[0.5, 0.25, 0.25])  # **调整任务类型比例**

            # 高优先级任务集中在指定区域
            if task_type == 'high':
                # 在两个高优先级区域均分任务
                if i < high_priority_task_count // 2:
                    selected_area = high_priority_areas[0]  # 第一个区域
                else:
                    selected_area = high_priority_areas[1]  # 第二个区域

                x = np.random.normal(loc=selected_area[0], scale=GROUND_LENGTH * 0.1)
                y = np.random.normal(loc=selected_area[1], scale=GROUND_WIDTH * 0.05)
            # 中等优先级任务集中在另一个区域
            elif task_type == 'medium':
                x = np.random.normal(loc=medium_priority_area[0], scale=GROUND_LENGTH * 0.05)
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
        计算本地计算延迟，仅返回计算结果。
        """
        local_data = (1 - self.offload_fraction) * self.data_size  # bits
        local_cycles = local_data * self.cpu_cycles  # cycles
        local_computation_delay = local_cycles / max(self.local_compute_power, 1e-9)  # 增加防护
        return local_computation_delay

    def offload_to_uav(self, uav, compute_alloc, bandwidth_alloc):
        """
        计算任务卸载到 UAV 的时延。
        """
        if compute_alloc <= 0 or bandwidth_alloc <= 0 or self.offload_fraction <= 0:
            raise ValueError("Invalid resource allocation: compute_alloc or bandwidth_alloc is zero.")

        off_data = self.offload_fraction * self.data_size  # bits
        off_cycles = off_data * self.cpu_cycles  # cycles
        transmission_rate, snr = calculate_transmission_rate(
            device_position=self.position,
            uav_position=uav.position,
            bandwidth_per_user=bandwidth_alloc * uav.bandwidth
        )

        if transmission_rate <= 0:
            raise ValueError(f"Invalid transmission rate: {transmission_rate}, SNR: {snr}")

        transmission_delay = off_data / transmission_rate  # seconds
        compute_time = off_cycles / max(uav.compute_power * compute_alloc, 1e-9)  # seconds
        offload_delay = transmission_delay + compute_time
        return offload_delay, snr

    def calculate_total_delay(self, uav=None, compute_alloc=0.0, bandwidth_alloc=0.0):
        """
        计算任务的总时延，综合考虑本地计算和卸载计算。
        """
        # 本地计算时延
        local_computation_delay = self.compute_locally()

        # 如果没有有效的 UAV 分配或分配资源为零，则直接使用本地计算时延
        if uav is None or compute_alloc <= 0 or bandwidth_alloc <= 0 or self.offload_fraction <= 0:
            self.total_delay = local_computation_delay
            self.is_task_completed = self.total_delay <= self.max_delay
            return self.total_delay

        try:
            # 计算卸载任务的传输时延和 UAV 计算时延
            uav_computation_delay, snr = self.offload_to_uav(uav, compute_alloc, bandwidth_alloc)
        except ValueError as e:
            # 如果传输速率无效，则回退到本地计算
            print(f"[Error in offloading] {e}. Falling back to local computation.")
            self.total_delay = local_computation_delay
            self.is_task_completed = self.total_delay <= self.max_delay
            return self.total_delay

        # 剩余任务的本地计算时延
        remaining_local_data = (1 - self.offload_fraction) * self.data_size
        remaining_local_cycles = remaining_local_data * self.cpu_cycles
        remaining_local_computation_delay = remaining_local_cycles / max(self.local_compute_power, 1e-9)

        # 总时延是 UAV 部分时延和本地剩余时延的最大值
        self.total_delay = max(uav_computation_delay, remaining_local_computation_delay)
        # print(f"self.offload_fraction: {self.offload_fraction}")
        # print(f"uav_computation_delay: {uav_computation_delay}, remaining_local_computation_delay: {remaining_local_computation_delay}")
        # print(f"self.total_delay: {self.total_delay}")

        # 判断任务是否完成
        self.is_task_completed = self.total_delay <= self.max_delay
        return self.total_delay

    @staticmethod
    def plot_iot_devices(devices, uavs=None, title=None):
        """
        绘制 IoT 设备的分布图，并可选绘制 UAV 的位置和覆盖范围。
        未被任何 UAV 覆盖的设备将以不同的颜色或形状表示。

        参数：
        - devices: IoTDevice 对象的列表
        - uavs: UAV 对象的列表，可选
        - title: 字符串，图表的标题，可选
        """
        plt.figure(figsize=(8, 8))
        colors = {'low': 'green', 'medium': 'red', 'high': 'red'}
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
            heights = [uav.position[2] for uav in uavs]
            max_height = max(heights)
            min_height = min(heights)

            # 使用颜色渐变表示高度
            cmap = plt.get_cmap('plasma')
            norm = plt.Normalize(min_height, max_height)
            fig, ax = plt.subplots()

            for idx, uav in enumerate(uavs):
                plt.scatter(uav.position[0], uav.position[1],
                            c=[cmap(norm(uav.position[2]))],
                            marker='^', s=100, edgecolors='k', label='UAV' if idx == 0 else "")
                circle = plt.Circle((uav.position[0], uav.position[1]), uav.coverage_radius,
                                    color=cmap(norm(uav.position[2])), fill=False,
                                    linestyle='--', alpha=0.3)
                plt.gca().add_patch(circle)
                # 添加高度注释
                plt.text(uav.position[0], uav.position[1], f"{uav.position[2]}m",
                         fontsize=9, ha='center', va='bottom')

            # 添加颜色条以表示高度
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
            cbar.set_label('UAV 高度 (m)')

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
    def __init__(self, uav_id, position):
        """
        初始化 UAV 对象，包括 UAV ID、三维位置等参数。
        """
        self.energy_consumed = 0.0
        self.uav_id = uav_id
        self.position = np.array(position)  # UAV 的三维位置 (x, y, z)
        self.coverage_angle = UAV_COVERAGE_ANGLE  # 覆盖角度，单位：度
        self.coverage_radius = self.calculate_coverage_radius()  # 根据高度计算覆盖半径
        self.compute_power = UAV_CPU_FREQUENCY
        self.bandwidth = UAV_BANDWIDTH
        self.connected_devices = []  # 已连接的设备列表
        self.battery_energy = UAV_BATTERY_ENERGY  # 电池能量
        self.min_energy = UAV_MIN_ENERGY  # 最小剩余能量
        self.speed = UAV_SPEED  # 飞行速度
        self.compute_allocation = {}  # 初始化计算资源分配策略
        self.bandwidth_allocation = {}  # {device_id: bandwidth_alloc}
        # 添加可用资源属性
        self.available_compute_resource = self.compute_power
        # 新增属性用于本step的关联请求收集
        self.candidate_associations = {}
        self.candidate_compute_allocation = {}
        self.candidate_offload_fraction = {}

    def calculate_coverage_radius(self):
        half_angle_rad = np.deg2rad(self.coverage_angle / 2)
        coverage_radius = self.position[2] * np.tan(half_angle_rad)
        return coverage_radius

    def reset(self):
        self.battery_energy = UAV_BATTERY_ENERGY
        self.available_compute_resource = self.compute_power
        self.compute_allocation = {}
        self.bandwidth_allocation = {}
        self.connected_devices = []
        self.energy_consumed = 0.0  # 重置能耗

        # 重置本step关联请求
        self.candidate_associations = {}
        self.candidate_compute_allocation = {}
        self.candidate_offload_fraction = {}

    def compute_energy_consumption(self):
        """
        计算 UAV 的能耗。
        返回值：total_energy: float，总能耗（焦耳）
        """
        total_energy = 0.0
        for device in self.connected_devices:
            offload_fraction = device.offload_fraction
            if offload_fraction > 0:
                # 计算卸载部分的能耗
                compute_alloc = self.compute_allocation.get(device.device_id, 0.0)
                if compute_alloc <= 0.0:
                    continue  # 避免无效的资源分配
                effective_compute_power = self.compute_power * compute_alloc
                off_cycles = offload_fraction * device.data_size * device.cpu_cycles
                compute_time = off_cycles / max(effective_compute_power, 1e-6)
                energy = UAV_HARDWARE_CONSTANT * (effective_compute_power ** 2) * compute_time
                total_energy += energy

        self.energy_consumed = total_energy  # 更新能耗属性
        return total_energy

    def apply_action(self, action, coverage_devices):
        """
        执行 UAV 的动作，包括设备关联策略、任务卸载比例和计算资源分配比例。
        """
        num_cov = len(coverage_devices)
        associations = action[0:num_cov]
        offloads = action[num_cov:2 * num_cov]
        allocs = action[2 * num_cov:3 * num_cov]

        # Reset previous candidate associations and allocations
        self.candidate_associations = {}
        self.candidate_compute_allocation = {}
        self.candidate_offload_fraction = {}

        for i, device in enumerate(coverage_devices):
            association = associations[i]
            offload_fraction = offloads[i]
            compute_alloc = allocs[i]

            if association == 1.0 and offload_fraction > 0.0:  # Using a threshold to decide on association (binary decision)
                self.candidate_associations[device.device_id] = True
                self.candidate_offload_fraction[device.device_id] = offload_fraction
                self.candidate_compute_allocation[device.device_id] = compute_alloc

                # Set device's offload_fraction
                device.offload_fraction = offload_fraction
            else:
                self.candidate_associations[device.device_id] = False
                device.offload_fraction = 0.0

    def get_observation(self, coverage_devices):
        """
        返回 UAV 的观测，包括自身状态和覆盖范围内设备的信息。
        """
        # UAV状态: [available_compute_resource, battery_energy]
        # UAV状态
        uav_info = np.array([
            self.available_compute_resource,
            self.battery_energy
        ], dtype=np.float32)

        # 设备信息：取覆盖范围内前MAX_DEVICES个设备信息
        devices_info = []
        for device in coverage_devices[:MAX_DEVICES]:
            device_info = [
                device.data_size,
                device.cpu_cycles,
                device.max_delay,
            ]
            devices_info.extend(device_info)

        # 如果覆盖设备不足MAX_DEVICES，填充0
        num_padded = MAX_DEVICES - len(coverage_devices)
        for _ in range(num_padded):
            devices_info.extend([0.0, 0.0, 0.0])

        devices_info = np.array(devices_info, dtype=np.float32)

        return np.concatenate([uav_info, devices_info])


class MultiUAVEnv:
    def __init__(self, K=K, min_threshold=CMIN, max_threshold=CMAX, alpha=0.5, beta=0.5, max_iterations=100):
        """
        初始化 MultiUAVEnv
        """
        # 1. 生成 IoT 设备
        self.devices = IoTDevice.generate_iot_devices(NUM_IOTS)

        # 2. UAV初始化(包括K-means,省略)
        clustering = ImprovedKMeansClustering(
            devices=self.devices,
            K=K,
            H_initial=INITIAL_HEIGHT,
            H_min=MIN_HEIGHT,
            H_max=MAX_HEIGHT,
            coverage_angle=UAV_COVERAGE_ANGLE,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            max_iterations=max_iterations,
            alpha=alpha,
            beta=beta
        )
        uav_positions, uav_heights = clustering.fit()

        self.uavs = {}
        for idx in range(K):
            uav_id = f'uav_{idx}'
            position_3d = (uav_positions[idx][0], uav_positions[idx][1], uav_heights[idx])
            uav = UAV(uav_id=uav_id, position=position_3d)
            self.uavs[uav_id] = uav

        # 可视化初始位置
        self.render(title="初始 UAV 部署位置和高度")

        # 3. 定义观察/动作空间
        obs_dim = 2 + MAX_DEVICES * 3
        action_dim = 3 * MAX_DEVICES
        self.action_spaces = {}
        self.observation_spaces = {}
        for uav_id in self.uavs.keys():
            self.observation_spaces[uav_id] = spaces.Box(
                low=0.0, high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32
            )
            # Adjust action space to accommodate binary association and bounded continuous values
            self.action_spaces[uav_id] = spaces.Box(
                low=0.0, high=1.0,
                shape=(action_dim,),
                dtype=np.float32
            )
        self.observation_space = spaces.Dict(self.observation_spaces)
        self.action_space = spaces.Dict(self.action_spaces)

        self.time_step = 0

    def reset(self):
        for device in self.devices:
            device.reset()
        for uav in self.uavs.values():
            uav.reset()
        self._calculate_device_coverage()

        observations = {}
        for uav_id, uav in self.uavs.items():
            coverage_devices = sorted([d for d in self.devices if uav in d.covering_uavs], key=lambda d: d.device_id)
            observations[uav_id] = uav.get_observation(coverage_devices)

        self.time_step = 0
        return observations

    def _calculate_device_coverage(self):
        for device in self.devices:
            device.covering_uavs = []
            for uav in self.uavs.values():
                dist = np.linalg.norm(device.position - uav.position[:2])
                if dist <= uav.coverage_radius:
                    device.covering_uavs.append(uav)
            device.is_within_coverage = bool(device.covering_uavs)

    def resolve_associations(self):
        """
        最终分配：每个device选1个UAV(最小时延), UAV内部对alloc做再次归一化以确保总和<=1
        """
        # 先清空connected_devices
        for uav in self.uavs.values():
            uav.connected_devices = []
            uav.bandwidth_allocation = {}
            uav.compute_allocation = {}

        devices_sorted = sorted(self.devices, key=lambda d: -d.priority)
        for device in devices_sorted:
            if device.is_task_completed:
                continue

            # 找出assoc=1的UAV
            candidate_uavs = [uav for uav in device.covering_uavs if
                              uav.candidate_associations.get(device.device_id, False)]
            if not candidate_uavs:
                # 无候选 => local
                device.connected_uav = None
                device.offload_fraction = 0.0
                continue

            best_uav = None
            min_delay = float('inf')
            best_alloc = 0.0
            best_offload = 0.0

            # 选时延最小的UAV
            for uav in candidate_uavs:
                offload_fraction = uav.candidate_offload_fraction.get(device.device_id, 0.0)
                compute_alloc = uav.candidate_compute_allocation.get(device.device_id, 0.0)
                # 临时假设connected_devices + [device]
                tmp_devs = uav.connected_devices + [device]
                num_conn = len(tmp_devs)
                bw = (uav.bandwidth / num_conn) if num_conn > 0 else 0.0
                delay = device.calculate_total_delay(uav, compute_alloc, bw)
                if delay < min_delay:
                    min_delay = delay
                    best_uav = uav
                    best_alloc = compute_alloc
                    best_offload = offload_fraction

            if best_uav:
                device.connected_uav = best_uav
                device.offload_fraction = best_offload
                best_uav.connected_devices.append(device)
                best_uav.compute_allocation[device.device_id] = best_alloc
            else:
                device.connected_uav = None
                device.offload_fraction = 0.0

        # 分配带宽(平均)
        for uav_id, uav in self.uavs.items():
            num_devs = len(uav.connected_devices)
            if num_devs > 0:
                avg_bw = uav.bandwidth / num_devs
                for dev in uav.connected_devices:
                    uav.bandwidth_allocation[dev.device_id] = avg_bw
            else:
                uav.bandwidth_allocation = {}

        # **对每个UAV再对compute_alloc做一次归一化**，防止sum>1
        for uav_id, uav in self.uavs.items():
            sum_alloc = sum(uav.compute_allocation.get(dev.device_id, 0.0) for dev in uav.connected_devices)
            if sum_alloc > 1.0:
                for dev in uav.connected_devices:
                    old_alloc = uav.compute_allocation[dev.device_id]
                    new_alloc = old_alloc / (sum_alloc + 1e-9)
                    uav.compute_allocation[dev.device_id] = new_alloc

    def step(self, actions):
        self._calculate_device_coverage()

        # 1. UAV apply_action => candidate_associations
        for uav_id, action in actions.items():
            uav = self.uavs[uav_id]
            coverage_devices = sorted([d for d in self.devices if uav in d.covering_uavs], key=lambda d: d.device_id)
            uav.apply_action(action, coverage_devices)

        # 2. 冲突消解 resolve
        self.resolve_associations()

        # 打印debug信息
        for uav_id, uav in self.uavs.items():
            num_conn = len(uav.connected_devices)
            sum_alloc = sum(uav.compute_allocation.get(dev.device_id, 0.0) for dev in uav.connected_devices)
            # print(f"[DEBUG] UAV {uav_id} 关联的 IoT 设备数量: {num_conn}")
            # print(f"[DEBUG] UAV {uav_id} 分配的计算资源比例总和: {sum_alloc:.2f}")

        # 3. 最终计算时延
        for device in self.devices:
            if device.is_task_completed:
                continue
            if device.connected_uav:
                uav = device.connected_uav
                comp_alloc = uav.compute_allocation.get(device.device_id, 0.0)
                bw_alloc = uav.bandwidth_allocation.get(device.device_id, 0.0)
                device.calculate_total_delay(uav, comp_alloc, bw_alloc)
            else:
                # local
                device.calculate_total_delay()

        # 4. UAV能耗
        for uav in self.uavs.values():
            uav.compute_energy_consumption()

        # 5. reward
        rewards = {}
        for uav_id, uav in self.uavs.items():
            # Modified reward computation
            rewards[uav_id] = self.compute_reward(uav)

        # 6. new obs
        observations = {}
        for uav_id, uav in self.uavs.items():
            coverage_devices = sorted([d for d in self.devices if uav in d.covering_uavs], key=lambda d: d.device_id)
            observations[uav_id] = uav.get_observation(coverage_devices)

        self.time_step += 1

        all_tasks_done = all(d.is_task_completed for d in self.devices)
        dones = {uav_id: False for uav_id in self.uavs.keys()}
        truncs = {uav_id: False for uav_id in self.uavs.keys()}
        if self.time_step >= MAX_TIME_STEPS or all_tasks_done:
            dones = {uav_id: True for uav_id in self.uavs.keys()}
            truncs = {uav_id: self.time_step >= MAX_TIME_STEPS for uav_id in self.uavs.keys()}
        infos = {uav_id: {} for uav_id in self.uavs.keys()}

        return observations, rewards, dones, truncs, infos

    def compute_reward(self, uav):
        # 定义延迟满意度和能耗的权重
        w_delay = 1.0  # 延迟满意度的权重
        w_energy = 0.0001  # 能耗的权重，根据实际情况可能需要调整

        # 计算所有连接设备的总延迟满意度
        total_delay_satisfaction = 0.0
        for device in uav.connected_devices:
            delay_satisfaction = (device.max_delay - device.total_delay) / device.max_delay
            delay_satisfaction = max(delay_satisfaction, 0.0)
            total_delay_satisfaction += delay_satisfaction

        # 计算平均延迟满意度
        if len(uav.connected_devices) > 0:
            average_delay_satisfaction = total_delay_satisfaction / len(uav.connected_devices)
        else:
            average_delay_satisfaction = 0.0

        # 计算能耗惩罚
        energy_penalty = w_energy * uav.energy_consumed

        # 组合奖励
        reward = (w_delay * average_delay_satisfaction) - energy_penalty

        # 确保奖励不为负
        reward = max(reward, 0.0)

        # # 打印详细的奖励组成部分（用于调试）
        # print(f"[DEBUG] UAV {uav.uav_id} - Avg Delay Satisfaction: {average_delay_satisfaction:.4f}, "
        #       f"Energy Consumed: {uav.energy_consumed:.4f}, Reward: {reward:.4f}")

        return reward

    def render(self, title=None):
        uav_objects = list(self.uavs.values())
        IoTDevice.plot_iot_devices(self.devices, uavs=uav_objects, title=title)


def calculate_transmission_rate(device_position, uav_position, bandwidth_per_user):
    """
    计算 IoT 设备与无人机之间的传输速率。
    返回值：
    - transmission_rate: float, 传输速率（bps）
    """
    # 提取 UAV 的高度
    uav_height = uav_position[2]

    # 计算 IoT 设备与 UAV 之间的距离和水平距离
    device_position = np.array(device_position)
    uav_position_2d = np.array(uav_position[:2])  # 只取 x 和 y

    horizontal_distance = np.linalg.norm(device_position - uav_position_2d)
    distance = np.sqrt(horizontal_distance ** 2 + uav_height ** 2)

    # 计算 IoT 设备与 UAV 之间的仰角
    if horizontal_distance == 0:
        horizontal_distance = 1e-6  # 防止除以零
    theta = np.arctan(uav_height / horizontal_distance)

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

    return transmission_rate, snr
