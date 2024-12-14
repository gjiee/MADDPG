import numpy as np
import matplotlib.pyplot as plt
from utils import *

# 设置绘图的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

maddpg_scores = np.load('../data/my_maddpg_scores.npy')
maddpg_steps = np.load('../data/my_maddpg_steps.npy')

kmeans_scores = np.load('../data/kmeans_maddpg_scores.npy')
kmeans_steps = np.load('../data/kmeans_maddpg_steps.npy')



plot_learning_curve(x=maddpg_steps,
                    scores=(maddpg_scores, kmeans_scores),
                    filename='../plots/my_maddpg_vs_kmeans_maddpg.png')
