GROUND_LENGTH = 1000.0  # 模拟区域的长度（米）
GROUND_WIDTH = 1000.0   # 模拟区域的宽度（米）
UAV_HEIGHT = 100.0      # 无人机的飞行高度（米）
UAV_COVER = 220.0    # 无人机的覆盖半径（米）

# IoT
NUM_IOTS = 150
IOT_COMPUTE_POWER_MIN = 0.5e9
IOT_COMPUTE_POWER_MAX = 1.0e9

# UAV
UAV_CPU_FREQUENCY = 20.0e9      # 无人机的计算频率（20.0 GHz）
UAV_BATTERY_ENERGY = 5.0e6      # 无人机的电池能量（5 MJ）
UAV_MIN_ENERGY = 1.0e6          # 无人机的最小剩余能量阈值（1 MJ）
UAV_SPEED = 15.0                # 无人机的飞行速度（15 m/s）
UAV_BANDWIDTH = 40e6            # 无人机的总带宽（40 MHz）
UAV_MASS = 15.0                 # 无人机质量（kg）
HARDWARE_CONSTANT = 1e-27       # 硬件相关常数
ENERGY_PENALTY_FACTOR = 0.01  # 能耗方差惩罚因子，可根据实验进行调整
MAX_DEVICES = 25
MAX_DEVICES_KMEANS = 30
MAX_DEVICES_RANDOM = 40

MIN_ALLOCATION_THRESHOLD = 0.0001  # 最小资源分配阈值

# 添加硬件相关常数
UAV_HARDWARE_CONSTANT = 1e-18  # 根据需要调整
MAX_UAV_MOVE = 10.0  # UAV 每次移动的最大距离（米）

# 通信参数
CARRIER_FREQUENCY = 2e9         # 载波频率（2 GHz）
SPEED_OF_LIGHT = 3e8            # 光速（3e8 m/s）
NOISE_POWER_DENSITY = 6e-21     # 噪声功率谱密度（W/Hz）
IOT_TX_POWER = 0.1              # IoT 设备的发射功率（0.1 W）

Energy_WEIGHT = 0.001

MAX_TIME_STEPS = 100  # 设置合理的最大时间步数
UAV_MIN_ENERGY = 10  # 设置 UAV 的最小电池能量阈值

# 聚类算法参数
ISODATA_INITIAL_CLUSTERS = 2
ISODATA_MAX_CLUSTERS = 30
ISODATA_MIN_SAMPLES_SPLIT = 20   # 分裂的最小样本数
ISODATA_MAX_SAMPLES_MERGE = 15  # 合并的最大样本数
MERGE_DISTANCE_THRESHOLD = 220.0