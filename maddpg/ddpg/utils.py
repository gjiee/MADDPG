import numpy as np
import matplotlib.pyplot as plt

# def plot_iot_devices(devices, uavs=None, title=None):
#     """
#     绘制 IoT 设备的分布图，并可选绘制 UAV 的位置和覆盖范围。
#     未被任何 UAV 覆盖的设备将以不同的颜色或形状表示。
#
#     参数：
#     - devices: IoTDevice 对象的列表
#     - uavs: UAV 位置的列表，可选
#     - title: 字符串，图表的标题，可选
#     """
#     plt.figure(figsize=(8, 8))
#     colors = {'low': 'green', 'medium': 'orange', 'high': 'red'}
#     labels = {'low': '低优先级任务', 'medium': '中优先级任务', 'high': '高优先级任务'}
#
#     # test
#     # for device in devices:
#     #     print(f"device.is_covered:{device.is_covered}")
#
#     # 绘制被覆盖的设备
#     for task_type in ['low', 'medium', 'high']:
#         xs = [device.position[0] for device in devices if device.task_type == task_type and device.is_covered]
#         ys = [device.position[1] for device in devices if device.task_type == task_type and device.is_covered]
#         plt.scatter(xs, ys, c=colors[task_type], label=labels[task_type], alpha=0.6)
#
#     # 绘制未被覆盖的设备
#     xs_uncovered = [device.position[0] for device in devices if not device.is_covered]
#     ys_uncovered = [device.position[1] for device in devices if not device.is_covered]
#     if xs_uncovered:
#         plt.scatter(xs_uncovered, ys_uncovered, c='gray', label='未覆盖设备', marker='x', alpha=0.6)
#
#         # 绘制 UAV 位置和覆盖范围
#     if uavs is not None and len(uavs) > 0:
#         for idx, uav_position in enumerate(uavs):
#             plt.scatter(uav_position[0], uav_position[1], c='blue', marker='^', s=100, label='UAV' if idx == 0 else "")
#             circle = plt.Circle((uav_position[0], uav_position[1]), UAV_COVER, color='blue', fill=False, linestyle='--',
#                                 alpha=0.3)
#             plt.gca().add_patch(circle)
#
#     if title is not None:
#         plt.title(title)
#     else:
#         plt.title('IoT 设备分布图及 UAV 覆盖范围')
#     plt.xlabel('X 轴位置 (m)')
#     plt.ylabel('Y 轴位置 (m)')
#     plt.xlim(-100, GROUND_LENGTH + 100)
#     plt.ylim(-100, GROUND_WIDTH + 100)
#     plt.legend()
#     plt.grid(True)
#     plt.show()

def plot_learning_curve(x, scores, filename, lines=None):
    maddpg_scores = scores

    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    # ax2 = fig.add_subplot(111, label="2", frame_on=False)

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

    # N = len(ddpg_scores)
    # running_avg = np.empty(N)
    # for t in range(N):
    #     running_avg[t] = np.mean(ddpg_scores[max(0, t-100):(t+1)])
    #
    # ax2.plot(x, running_avg, color="C1")
    # ax2.axes.get_xaxis().set_visible(False)
    # ax2.yaxis.tick_right()
    # ax2.set_ylabel('DDPG Score', color="C1")
    # ax2.yaxis.set_label_position('right')
    # ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

    # 显示图像
    plt.show()  # 确保调用了plt.show()来显示图像

