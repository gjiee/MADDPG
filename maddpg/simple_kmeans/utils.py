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
    ax.set_ylabel("MY_MADDPG Score", color="C0")
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
    ax2.set_ylabel('KMEANS_MADDPG Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

    # 显示图像
    plt.show()  # 确保调用了plt.show()来显示图像
