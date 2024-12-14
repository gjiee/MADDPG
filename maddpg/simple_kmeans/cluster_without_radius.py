# cluster.py

'''
ISODATA：用于确定最佳的聚类数量和初始的聚类中心
ISODATA_KMeans_Clustering：进一步优化 UAV 的位置，使得覆盖率和负载均衡达到最优
'''

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from parameters import *
from env import *

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
