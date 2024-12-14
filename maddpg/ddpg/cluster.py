# cluster.py

'''
ISODATA：用于确定最佳的聚类数量和初始的聚类中心
ISODATA_KMeans_Clustering：进一步优化 UAV 的位置，使得覆盖率和负载均衡达到最优
'''

from parameters import *
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np

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
    加权 KMeans 聚类进一步优化 UAV 的位置，通过设备任务优先级作为权重优化聚类，使得每个 UAV 覆盖率更高，负载方差更小。
    assign_devices_to_uavs:将设备分配到最近的 UAV，考虑覆盖范围。
    evaluate_solution:计算覆盖率和负载方差
    update_centroids:基于分配的iot设备和任务权重更新聚类中心,并且利用DDPG的能耗方差反馈优化UAV位置部署
    '''
    def __init__(self, devices, fixed_k, initial_centroids, max_iter=100):
        """
        使用 ISODATA 确定的聚类数量和初始聚类中心初始化 ISODATA_KMeans_Clustering
        """
        self.devices = devices  # 设备列表
        self.data = np.array([device.position for device in devices])  # 设备位置的二维数组
        self.weights = np.array([device.priority for device in devices])  # 设备的任务优先级
        self.fixed_k = fixed_k  # 固定的聚类数
        self.initial_centroids = initial_centroids  # 初始聚类中心
        self.max_iter = max_iter  # 最大迭代次数

    def fit(self):
        # 使用固定的 K 和初始聚类中心
        centroids = self.initial_centroids.copy()
        K = self.fixed_k
        best_score = -np.inf  # 初始化最佳得分
        best_uav_positions = None
        best_device_assignments = None

        # 覆盖率与负载方差的权重系数
        gamma_coverage = 0.5  # 覆盖率的权重
        gamma_variance = 0.5  # 负载方差的权重

        for iteration in range(1, self.max_iter + 1):
            # 动态调整权重
            gamma_coverage = gamma_coverage * (1 - 0.05 * iteration)  # 随迭代次数略微增加覆盖率权重
            gamma_variance = gamma_variance * (1 + 0.05 * iteration)  # 随迭代次数略微减少负载方差权重

            # 执行聚类与设备分配
            kmeans = KMeans(n_clusters=K, init=centroids, n_init=1, max_iter=100)
            clusters = kmeans.fit_predict(self.data, sample_weight=self.weights)
            centroids = kmeans.cluster_centers_

            device_assignments = self.assign_devices_to_uavs(centroids)
            coverage_rate, load_variance = self.evaluate_solution(device_assignments)

            # 动态得分计算并记录
            score = gamma_coverage * coverage_rate - gamma_variance * load_variance
            if score > best_score:
                best_score = score
                best_uav_positions = centroids.copy()
                best_device_assignments = device_assignments.copy()

                # 检查是否提前停止
                if coverage_rate >= 0.99 and load_variance <= 0.05:
                    print("达到接近最优解，提前结束迭代。")
                    break

        # 最终的设备分配和 UAV 位置
        self.uav_positions = best_uav_positions
        self.device_assignments = best_device_assignments

        return self.device_assignments, self.uav_positions

    def assign_devices_to_uavs(self, centroids):
        """
        分配设备到 UAV，考虑覆盖范围。如果设备在多个 UAV 的覆盖范围内，分配给最近的 UAV。
        同时更新设备的 is_covered 属性。
        """
        device_assignments = [[] for _ in range(len(centroids))]
        for device in self.devices:
            distances = np.linalg.norm(centroids - device.position, axis=1)
            within_coverage = np.where(distances <= UAV_COVER)[0]

            if len(within_coverage) > 0:
                # 分配给最近的 UAV
                nearest_uav = within_coverage[np.argmin(distances[within_coverage])]
                device.cluster_label = nearest_uav
                device.is_covered = True  # 确保设备被正确标记为被覆盖
                device_assignments[nearest_uav].append(device)
            else:
                # 设备未被任何 UAV 覆盖
                device.cluster_label = -1
                device.is_covered = False  # 明确标记设备未被覆盖
        return device_assignments

    def evaluate_solution(self, device_assignments):
        """
        计算覆盖率和负载方差。
        """
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

