# cluster.py

'''
ISODATA：用于确定最佳的聚类数量和初始的聚类中心
ISODATA_KMeans_Clustering：进一步优化 UAV 的位置，使得覆盖率和负载均衡达到最优
'''

from parameters import *
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.metrics import pairwise_distances

class ISODATA:
    '''
    ISODATA 聚类用于确定初始聚类数量和中心位置，通过分裂和合并调整聚类数量，确保合理的初始 UAV 数量和位置。
    '''
    def __init__(self, devices, initial_clusters=ISODATA_INITIAL_CLUSTERS, max_clusters=ISODATA_MAX_CLUSTERS,
                 min_samples_split=ISODATA_MIN_SAMPLES_SPLIT, max_samples_merge=ISODATA_MAX_SAMPLES_MERGE,
                 max_iter=100):
        self.devices = devices
        self.data = np.array([device.position for device in devices])
        self.initial_clusters = initial_clusters
        self.max_clusters = max_clusters
        self.min_samples_split = min_samples_split
        self.max_samples_merge = max_samples_merge
        self.max_iter = max_iter

    def fit(self):
        # 初始 KMeans 聚类
        kmeans = KMeans(n_clusters=self.initial_clusters, max_iter=300, n_init=10)
        clusters = kmeans.fit_predict(self.data)
        centroids = kmeans.cluster_centers_

        for iteration in range(self.max_iter):
            prev_clusters = clusters.copy()
            clusters, centroids = self.split_and_merge(clusters, centroids)
            if np.array_equal(clusters, prev_clusters):
                break  # 聚类结果不再变化，停止迭代

        # 重新映射聚类标签
        unique_labels = np.unique(clusters)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        clusters = np.array([label_mapping[label] for label in clusters])
        centroids = centroids[unique_labels]

        # 构建簇列表
        clusters_list = [[] for _ in range(len(unique_labels))]
        for device, cluster_label in zip(self.devices, clusters):
            clusters_list[cluster_label].append(device)
            device.cluster_label = cluster_label

        return clusters_list, centroids

    def split_and_merge(self, clusters, centroids):
        new_clusters = clusters.copy()
        new_centroids = centroids.copy()
        current_k = len(new_centroids)

        # 分裂操作
        for i in range(current_k):
            cluster_indices = np.where(new_clusters == i)[0]
            cluster_data = self.data[cluster_indices]
            if len(cluster_data) > self.min_samples_split and current_k < self.max_clusters:
                # 对当前簇进行二次 KMeans 分裂
                kmeans_split = KMeans(n_clusters=2, max_iter=100, n_init=10)
                sub_labels = kmeans_split.fit_predict(cluster_data)
                sub_centroids = kmeans_split.cluster_centers_

                # 更新聚类标签
                max_label = new_clusters.max()
                sub_labels[sub_labels == 0] = i  # 保持原有标签
                sub_labels[sub_labels == 1] = max_label + 1  # 新的子簇标签

                new_clusters[cluster_indices] = sub_labels
                new_centroids[i] = sub_centroids[0]
                new_centroids = np.vstack([new_centroids, sub_centroids[1]])
                current_k += 1

        # 合并操作
        distances = cdist(new_centroids, new_centroids)
        np.fill_diagonal(distances, np.inf)

        merged_clusters = set()
        for i in range(len(new_centroids)):
            if i in merged_clusters:
                continue
            cluster_i_indices = np.where(new_clusters == i)[0]
            cluster_i_size = len(cluster_i_indices)
            if cluster_i_size <= self.max_samples_merge:
                j = np.argmin(distances[i])
                if j in merged_clusters:
                    continue
                cluster_j_indices = np.where(new_clusters == j)[0]
                cluster_j_size = len(cluster_j_indices)
                if cluster_j_size <= self.max_samples_merge:
                    # 合并簇 i 和簇 j
                    new_clusters[cluster_j_indices] = i
                    combined_indices = np.where(new_clusters == i)[0]
                    combined_data = self.data[combined_indices]
                    new_centroids[i] = np.mean(combined_data, axis=0)
                    merged_clusters.add(j)

        # 删除已合并的簇中心
        keep_indices = [idx for idx in range(len(new_centroids)) if idx not in merged_clusters]
        old_to_new_indices = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}
        new_centroids = new_centroids[keep_indices]

        updated_clusters = new_clusters.copy()
        for old_idx, new_idx in old_to_new_indices.items():
            updated_clusters[new_clusters == old_idx] = new_idx
        new_clusters = updated_clusters

        return new_clusters, new_centroids

class ISODATA_KMeans_Clustering:
    '''
    加权 KMeans 聚类进一步优化 UAV 的位置和高度，通过设备任务优先级作为权重优化聚类，使得每个 UAV 覆盖率更高，负载方差更小。
    '''
    def __init__(self, devices, fixed_k, initial_centroids, initial_heights, max_iter=100):
        self.devices = devices
        self.data = np.array([device.position for device in devices])
        self.weights = np.array([device.priority for device in devices])
        self.fixed_k = fixed_k
        self.initial_centroids = initial_centroids
        self.initial_heights = initial_heights
        self.max_iter = max_iter

    def adjust_coverage_radius(self, height):
        """根据高度计算覆盖半径"""
        BASE_RADIUS = 100  # 高度为 100 时的覆盖半径
        return BASE_RADIUS * (height / 100)

    def fit(self):
        centroids = self.initial_centroids.copy()
        heights = self.initial_heights.copy()
        K = self.fixed_k
        best_score = -np.inf
        best_uav_positions = None
        best_device_assignments = None
        best_heights = None

        gamma_coverage = 0.5
        gamma_variance = 0.5

        for iteration in range(1, self.max_iter + 1):
            gamma_coverage = gamma_coverage * (1 - 0.05 * iteration)
            gamma_variance = gamma_variance * (1 + 0.05 * iteration)

            # 执行聚类与设备分配
            kmeans = KMeans(n_clusters=K, init=centroids, n_init=1, max_iter=100)
            clusters = kmeans.fit_predict(self.data, sample_weight=self.weights)
            centroids = kmeans.cluster_centers_

            device_assignments = self.assign_devices_to_uavs(centroids)
            coverage_rate, load_variance = self.evaluate_solution(device_assignments)

            score = gamma_coverage * coverage_rate - gamma_variance * load_variance
            if score > best_score:
                best_score = score
                best_uav_positions = centroids.copy()
                best_device_assignments = device_assignments.copy()
                best_heights = [self.calculate_uav_height(len(devices)) for devices in device_assignments]

                if coverage_rate >= 0.99 and load_variance <= 0.05:
                    print("达到接近最优解，提前结束迭代。")
                    break

        self.uav_positions = best_uav_positions
        self.device_assignments = best_device_assignments
        self.uav_heights = best_heights

        return self.device_assignments, self.uav_positions, self.uav_heights

    def assign_devices_to_uavs(self, centroids):
        device_assignments = [[] for _ in range(len(centroids))]
        for device in self.devices:
            distances = np.linalg.norm(centroids - device.position, axis=1)
            within_coverage = [i for i, d in enumerate(distances) if d <= self.adjust_coverage_radius(self.initial_heights[i])]

            if within_coverage:
                nearest_uav = within_coverage[np.argmin([distances[i] for i in within_coverage])]
                device_assignments[nearest_uav].append(device)
            else:
                # 设备未被任何 UAV 覆盖，可以选择分配给最近的 UAV 或标记为未覆盖
                nearest_uav = np.argmin(distances)
                device_assignments[nearest_uav].append(device)

        return device_assignments

    def calculate_uav_height(self, device_count):
        MIN_HEIGHT = 50
        MAX_HEIGHT = 300
        k1, k2 = 5, 50
        return max(MIN_HEIGHT, min(MAX_HEIGHT, k1 * device_count + k2))

    def evaluate_solution(self, device_assignments):
        total_devices = len(self.devices)
        covered_devices = sum([len(devices) for devices in device_assignments])
        coverage_rate = covered_devices / total_devices

        # 计算权重方差（每个 UAV 服务的 IoT 设备的权重总和的方差）
        loads = np.array([sum(device.priority for device in devices) for devices in device_assignments])
        if len(loads) > 1:
            load_variance = np.var(loads)
        else:
            load_variance = 0.0

        return coverage_rate, load_variance

# class ImprovedKMeansClustering:
#     '''
#     基于改进 K-means 的多 UAV 部署优化算法
#     '''
#     def __init__(self, devices, K, H_initial=INITIAL_HEIGHT, H_min=MIN_HEIGHT, H_max=MAX_HEIGHT,
#                  coverage_angle=UAV_COVERAGE_ANGLE, min_threshold=CMIN, max_threshold=CMAX,
#                  max_iterations=100, alpha=0.5, beta=0.5):
#         """
#         初始化聚类算法的参数。
#
#         参数：
#             - devices: IoTDevice 对象的列表
#             - K: UAV 数量
#             - H_initial: UAV 初始高度
#             - H_min: UAV 最小高度
#             - H_max: UAV 最大高度
#             - coverage_angle: UAV 覆盖角度
#             - min_threshold: 最小设备数量阈值
#             - max_threshold: 最大设备数量阈值
#             - max_iterations: 最大迭代次数
#             - alpha: 覆盖率权重
#             - beta: 负载方差权重
#         """
#         self.devices = devices
#         self.K = K
#         self.H_initial = H_initial
#         self.H_min = H_min
#         self.H_max = H_max
#         self.coverage_angle = coverage_angle
#         self.min_threshold = min_threshold
#         self.max_threshold = max_threshold
#         self.max_iterations = max_iterations
#         self.alpha = alpha
#         self.beta = beta
#         self.weights = np.array([device.priority for device in devices])
#
#     def calculate_coverage_radius(self, height):
#         """根据高度计算覆盖半径"""
#         half_angle_rad = np.deg2rad(self.coverage_angle / 2)
#         coverage_radius = height * np.tan(half_angle_rad)
#         return coverage_radius
#
#     def fit(self):
#         # 提取设备的二维坐标
#         device_positions = np.array([device.position for device in self.devices])
#
#         # 初始 KMeans 聚类
#         kmeans = KMeans(n_clusters=self.K, init='k-means++', max_iter=300, n_init=10, random_state=42)
#         clusters = kmeans.fit_predict(device_positions)
#         centroids = kmeans.cluster_centers_
#
#         # 初始化 UAV 高度
#         heights = [self.H_initial for _ in range(self.K)]
#
#         # 记录最佳方案
#         best_score = np.inf
#         best_uav_positions = None
#         best_uav_heights = None
#
#         for iteration in range(1, self.max_iterations + 1):
#             # 设备分配
#             device_assignments = [[] for _ in range(self.K)]
#             for device in self.devices:
#                 device_position = device.position
#                 covering_uavs = []
#                 distances = []
#
#                 # 找出覆盖该设备的所有 UAV
#                 for uav_idx in range(self.K):
#                     uav_position = centroids[uav_idx]
#                     height = heights[uav_idx]
#                     coverage_radius = self.calculate_coverage_radius(height)
#                     distance = np.linalg.norm(device_position - uav_position)
#                     if distance <= coverage_radius:
#                         covering_uavs.append(uav_idx)
#                         distances.append(distance)
#
#                 if covering_uavs:
#                     # 如果有多个 UAV 覆盖，分配给最近的 UAV
#                     min_distance_idx = np.argmin(distances)
#                     selected_uav = covering_uavs[min_distance_idx]
#                     device_assignments[selected_uav].append(device)
#                 else:
#                     # 设备不在任何 UAV 的覆盖范围内，不分配，任务在本地计算
#                     pass  # 任务在本地计算，无需分配
#
#             # 计算指标
#             total_devices = len(self.devices)
#             covered_devices = sum([len(devices) for devices in device_assignments])
#             coverage_rate = covered_devices / total_devices
#
#             # 负载方差
#             loads = np.array([len(devices) for devices in device_assignments])
#             load_variance = np.var(loads)
#
#             # 归一化指标
#             cost_coverage = 1 - coverage_rate  # 覆盖率越大越好，成本越低
#             # 负载方差归一化（假设方差最大为某个合理值，如 CMAX^2）
#             cost_load_variance = load_variance / (self.K * (self.max_threshold - self.min_threshold) ** 2 + 1e-6)  # 避免除以零
#
#             # 计算综合评分
#             score = self.alpha * cost_coverage + self.beta * cost_load_variance
#
#             # 更新最佳方案
#             if score < best_score:
#                 best_score = score
#                 best_uav_positions = centroids.copy()
#                 best_uav_heights = heights.copy()
#
#             # 检查收敛条件
#             if coverage_rate >= 0.99 and load_variance <= 0.05:
#                 print(f"Early stopping at iteration {iteration} with Score: {score:.4f}")
#                 break
#
#             # 动态调整 UAV 高度
#             for uav_idx in range(self.K):
#                 n = len(device_assignments[uav_idx])
#                 if n < self.min_threshold:
#                     heights[uav_idx] = min(heights[uav_idx] + 10, self.H_max)
#                 elif n > self.max_threshold:
#                     heights[uav_idx] = max(heights[uav_idx] - 10, self.H_min)
#                 # else: 高度不变
#
#             # 更新 UAV 位置为已分配设备的质心
#             for uav_idx in range(self.K):
#                 if len(device_assignments[uav_idx]) > 0:
#                     new_x = np.mean([device.position[0] for device in device_assignments[uav_idx]])
#                     new_y = np.mean([device.position[1] for device in device_assignments[uav_idx]])
#                     centroids[uav_idx] = np.array([new_x, new_y])
#                 # else: 保持原位置
#
#         return best_uav_positions, best_uav_heights

class ImprovedKMeansClustering:
    '''
    基于改进 K-means 的多 UAV 部署优化算法
    '''
    def __init__(self, devices, K, H_initial=INITIAL_HEIGHT, H_min=MIN_HEIGHT, H_max=MAX_HEIGHT,
                 coverage_angle=UAV_COVERAGE_ANGLE, min_threshold=CMIN, max_threshold=CMAX,
                 max_iterations=100, alpha=0.5, beta=0.5):
        """
        初始化聚类算法的参数。

        参数：
            - devices: IoTDevice 对象的列表
            - K: UAV 数量
            - H_initial: UAV 初始高度
            - H_min: UAV 最小高度
            - H_max: UAV 最大高度
            - coverage_angle: UAV 覆盖角度
            - min_threshold: 最小设备数量阈值
            - max_threshold: 最大设备数量阈值
            - max_iterations: 最大迭代次数
            - alpha: 覆盖率权重
            - beta: 负载方差权重
        """
        self.devices = devices
        self.K = K
        self.H_initial = H_initial
        self.H_min = H_min
        self.H_max = H_max
        self.coverage_angle = coverage_angle
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.weights = np.array([device.priority for device in devices])

    def calculate_coverage_radius(self, height):
        """根据高度计算覆盖半径"""
        half_angle_rad = np.deg2rad(self.coverage_angle / 2)
        coverage_radius = height * np.tan(half_angle_rad)
        return coverage_radius

    def fit(self):
        # 提取设备的二维坐标
        device_positions = np.array([device.position for device in self.devices])

        # 初始 KMeans 聚类
        kmeans = KMeans(n_clusters=self.K, init='k-means++', max_iter=300, n_init=10, random_state=42)
        clusters = kmeans.fit_predict(device_positions)
        centroids = kmeans.cluster_centers_

        # 初始化 UAV 高度
        heights = [self.H_initial for _ in range(self.K)]

        # 记录最佳方案
        best_score = np.inf
        best_uav_positions = None
        best_uav_heights = None

        for iteration in range(1, self.max_iterations + 1):
            # 设备分配
            device_assignments = [[] for _ in range(self.K)]
            for device in self.devices:
                device_position = device.position
                covering_uavs = []
                distances = []

                # 找出覆盖该设备的所有 UAV
                for uav_idx in range(self.K):
                    uav_position = centroids[uav_idx]
                    height = heights[uav_idx]
                    coverage_radius = self.calculate_coverage_radius(height)
                    distance = np.linalg.norm(device_position - uav_position)
                    if distance <= coverage_radius:
                        covering_uavs.append(uav_idx)
                        distances.append(distance)

                if covering_uavs:
                    # 如果有多个 UAV 覆盖，分配给最近的 UAV
                    min_distance_idx = np.argmin(distances)
                    selected_uav = covering_uavs[min_distance_idx]
                    device_assignments[selected_uav].append(device)
                else:
                    # 设备不在任何 UAV 的覆盖范围内，不分配，任务在本地计算
                    pass  # 任务在本地计算，无需分配

            # 计算指标
            total_devices = len(self.devices)
            covered_devices = sum([len(devices) for devices in device_assignments])
            coverage_rate = covered_devices / total_devices

            # 负载方差
            loads = np.array([len(devices) for devices in device_assignments])
            load_variance = np.var(loads)

            # 归一化指标
            cost_coverage = 1 - coverage_rate  # 覆盖率越大越好，成本越低
            # 负载方差归一化（假设方差最大为某个合理值，如 CMAX^2）
            cost_load_variance = load_variance / (self.K * (self.max_threshold - self.min_threshold) ** 2 + 1e-6)  # 避免除以零

            # 计算综合评分
            score = self.alpha * cost_coverage + self.beta * cost_load_variance

            # 更新最佳方案
            if score < best_score:
                best_score = score
                best_uav_positions = centroids.copy()
                best_uav_heights = heights.copy()

            # 检查收敛条件
            if coverage_rate >= 0.99 and load_variance <= 0.05:
                print(f"Early stopping at iteration {iteration} with Score: {score:.4f}")
                break

            # 动态调整 UAV 高度
            for uav_idx in range(self.K):
                n = len(device_assignments[uav_idx])
                if n < self.min_threshold:
                    heights[uav_idx] = min(heights[uav_idx] + 10, self.H_max)
                elif n > self.max_threshold:
                    heights[uav_idx] = max(heights[uav_idx] - 10, self.H_min)
                # else: 高度不变

            # 更新 UAV 位置为已分配设备的质心
            for uav_idx in range(self.K):
                if len(device_assignments[uav_idx]) > 0:
                    new_x = np.mean([device.position[0] for device in device_assignments[uav_idx]])
                    new_y = np.mean([device.position[1] for device in device_assignments[uav_idx]])
                    centroids[uav_idx] = np.array([new_x, new_y])
                # else: 保持原位置

        return best_uav_positions, best_uav_heights




class SimpleKMeansClustering:
    """
    使用固定 K 值和经典 K-Means 聚类将 IoT 设备划分到簇。
    UAV 的位置被设置为每个簇的质心。
    不考虑覆盖范围和负载均衡。
    """
    def __init__(self, devices, fixed_k, max_iter=300, random_state=42):
        """
        初始化 SimpleKMeansClustering。

        参数：
        - devices: IoTDevice 对象的列表
        - num_clusters: int, 簇的数量（即 UAV 的数量）
        - max_iter: int, K-Means 的最大迭代次数
        - random_state: int, 随机数种子
        """
        self.devices = devices
        self.fixed_k = fixed_k
        self.max_iter = max_iter
        self.random_state = random_state
        self.data = np.array([device.position for device in devices])  # 提取设备位置

    def fit(self):
        """
        执行 K-Means 聚类，将 IoT 设备划分到簇并部署 UAV 到簇的质心。

        返回：
        - device_assignments: 每个簇分配的 IoTDevice 对象列表
        - uav_positions: UAV 的部署位置（质心）
        """
        kmeans = KMeans(n_clusters=self.fixed_k, max_iter=self.max_iter, random_state=self.random_state)
        clusters = kmeans.fit_predict(self.data)
        centroids = kmeans.cluster_centers_

        # 构建簇列表
        device_assignments = [[] for _ in range(self.fixed_k)]
        for device, cluster_label in zip(self.devices, clusters):
            device_assignments[cluster_label].append(device)
            device.cluster_label = cluster_label

        return device_assignments, centroids

class ISODATA_Random_UAV_Deployment:
    """
    通过 ISODATA 确定最佳的聚类数量后，从 IoT 设备的随机位置选择 K 个作为 UAV 的部署位置。
    """
    def __init__(self, devices, initial_clusters=ISODATA_INITIAL_CLUSTERS, max_iter=100, random_state=42):
        """
        初始化 ISODATA_Random_UAV_Deployment 类。

        参数：
        - devices: IoTDevice 对象的列表
        - initial_clusters: int, 初始聚类数量
        - max_iter: int, K-Means 聚类的最大迭代次数
        - random_state: int, 随机数种子
        """
        self.devices = devices
        self.initial_clusters = initial_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.data = np.array([device.position for device in devices])  # 提取设备位置

    def fit(self):
        """
        执行 ISODATA 聚类确定聚类数量，然后从设备中随机选择 K 个位置作为 UAV 部署位置。

        返回：
        - device_assignments: 每个簇分配的 IoTDevice 对象列表
        - uav_positions: 随机选择的 UAV 部署位置
        """
        # 使用 ISODATA 确定最佳的聚类数量
        isodata = ISODATA(self.devices, initial_clusters=self.initial_clusters, max_iter=self.max_iter)
        clusters_list, centroids = isodata.fit()

        # 根据聚类数量确定 UAV 的数量
        k = len(centroids)

        # 从 IoT 设备中随机选择 K 个位置作为 UAV 部署位置
        random_indices = np.random.choice(len(self.devices), k, replace=False)
        uav_positions = [self.devices[i].position for i in random_indices]

        # 将设备分配到最近的 UAV
        device_assignments = [[] for _ in range(k)]
        for device in self.devices:
            distances = [np.linalg.norm(device.position - np.array(uav_pos)) for uav_pos in uav_positions]
            closest_uav = np.argmin(distances)
            device_assignments[closest_uav].append(device)
            device.cluster_label = closest_uav  # 更新设备的簇标签

        return device_assignments, uav_positions