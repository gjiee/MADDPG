import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, filename, lines=None):
    # 解包 MADDPG 和 DDPG 的分数
    maddpg_scores, ddpg_scores = scores

    # 创建画布
    fig, ax = plt.subplots()

    # 计算 MADDPG 的移动平均分数
    N = len(maddpg_scores)
    running_avg_maddpg = np.empty(N)
    for t in range(N):
        running_avg_maddpg[t] = np.mean(
            maddpg_scores[max(0, t-100):(t+1)]
        )

    # 计算 DDPG 的移动平均分数
    N = len(ddpg_scores)
    running_avg_ddpg = np.empty(N)
    for t in range(N):
        running_avg_ddpg[t] = np.mean(
            ddpg_scores[max(0, t-100):(t+1)]
        )

    # 绘制曲线
    ax.plot(x, running_avg_maddpg, label="MADDPG", color="C0")
    ax.plot(x, running_avg_ddpg, label="DDPG", color="C1")

    # 添加标签和图例
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Scores")
    ax.legend(loc="upper left")

    # 添加垂直线（可选）
    if lines is not None:
        for line in lines:
            plt.axvline(x=line, linestyle="--", color="gray")

    # 保存并显示图像
    plt.savefig(filename)
    plt.show()
