import numpy as np
import matplotlib.pyplot as plt
from utils import *


# 设置绘图的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

maddpg_scores = np.load('../data/my_maddpg_scores.npy')
maddpg_steps = np.load('../data/my_maddpg_steps.npy')

# ddpg_scores = np.load('../data/my_ddpg_scores.npy')
# ddpg_steps = np.load('../data/my_ddpg_steps.npy')

plot_maddpg_learning_curve(x=maddpg_steps,
                    scores=(maddpg_scores),
                    filename='../plots/my_maddpg_1207')
# plot_learning_curve(x=maddpg_steps,
#                     scores=(maddpg_scores, ddpg_scores),
#                     filename='../plots/my_maddpg_vs_my_ddpg.png')
#
# kmeans_scores = np.load('../data/kmeans_maddpg_scores.npy')
# kmeans_steps = np.load('../data/kmeans_maddpg_steps.npy')
#
# random_scores = np.load('../data/random_maddpg_scores.npy')
# random_steps = np.load('../data/random_maddpg_steps.npy')
#
# plot_cluster_curve(x=maddpg_steps,
#                     scores=(maddpg_scores, kmeans_scores,random_scores),
#                     filename='../plots/my_vs_kmeans_vs_random.png')
