import numpy as np
import matplotlib.pyplot as plt
from utils import plot_learning_curve


# 设置绘图的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# maddpg_scores = np.load('../data/my_maddpg_scores.npy')
# maddpg_steps = np.load('../data/my_maddpg_steps.npy')

ddpg_scores = np.load('../data/my_ddpg_scores.npy')
ddpg_steps = np.load('../data/my_ddpg_steps.npy')

plot_learning_curve(x=ddpg_steps,
                    scores=(ddpg_scores),
                    filename='../plots/ddpg.png')

