import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, filename, lines=None):
    maddpg_scores = scores[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    N = len(maddpg_scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(
                maddpg_scores[max(0, t-100):(t+1)])

    ax.plot(x, running_avg, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("MADDPG Score", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    ddpg_scores = scores[1]

    N = len(ddpg_scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(ddpg_scores[max(0, t-100):(t+1)])

    ax2.plot(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('DDPG Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

    # 显示图像
    plt.show()  # 确保调用了plt.show()来显示图像

def plot_cluster_curve(x, scores, filename, lines=None):
    # scores[0] 是 MADDPG 曲线的得分
    my_scores = scores[0]
    # scores[1] 是 KMeans 曲线的得分
    kmeans_scores = scores[1]
    # scores[2] 是 Random 曲线的得分
    random_scores = scores[2]

    # 创建图形和三个子图
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111, label="1")
    ax2 = ax1.twinx()  # 第二个 y 轴
    ax3 = ax1.twinx()  # 第三个 y 轴

    # 调整 ax3 的位置，避免与其他 y 轴重叠
    ax3.spines['right'].set_position(('outward', 60))

    # 计算和绘制 MADDPG 曲线
    N = len(my_scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(my_scores[max(0, t - 100):(t + 1)])

    ax1.plot(x, running_avg, color="C0", label="My_Cluster", linewidth=2)
    ax1.set_xlabel("Training Steps", color="C0")
    ax1.set_ylabel("My_Cluster Score", color="C0")
    ax1.tick_params(axis='x', colors="C0")
    ax1.tick_params(axis='y', colors="C0")

    # 计算和绘制 KMeans 曲线
    running_avg = np.empty(len(kmeans_scores))
    for t in range(len(kmeans_scores)):
        running_avg[t] = np.mean(kmeans_scores[max(0, t - 100):(t + 1)])

    ax2.plot(x, running_avg, color="C1", label="KMeans", linewidth=2)
    ax2.set_ylabel('KMeans Score', color="C1")
    ax2.tick_params(axis='y', colors="C1")

    # 计算和绘制 Random 曲线
    running_avg = np.empty(len(random_scores))
    for t in range(len(random_scores)):
        running_avg[t] = np.mean(random_scores[max(0, t - 100):(t + 1)])

    ax3.plot(x, running_avg, color="C2", label="Random", linewidth=2)
    ax3.set_ylabel('Random Score', color="C2")
    ax3.tick_params(axis='y', colors="C2")

    # 如果提供了 lines，添加垂直线
    if lines is not None:
        for line in lines:
            plt.axvline(x=line, color="gray", linestyle="--", linewidth=1)

    # 添加图例
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax3.legend(loc="lower right")

    # 保存图像
    plt.savefig(filename)

    # 显示图像
    plt.show()

def plot_maddpg_learning_curve(x, scores, filename, lines=None):
    maddpg_scores = scores

    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")

    N = len(maddpg_scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(
                maddpg_scores[max(0, t-100):(t+1)])

    ax.plot(x, running_avg, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("MADDPG Score", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")


    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

    # 显示图像
    plt.show()  # 确保调用了plt.show()来显示图像