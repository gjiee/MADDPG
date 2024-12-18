# run.py
from buffer import MultiAgentReplayBuffer
from maddpg import MADDPG
import torch
import random
import numpy as np
from env import MultiUAVEnv
from parameters import *

'''
run：主要的训练循环，包含训练过程和评估过程。
evaluate：用于评估智能体的表现。
if __name__ == '__main__'：启动训练的代码块。
'''

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def evaluate(agents, env, ep, step, n_eval=5):
    '''
    evaluate 函数用于评估智能体的表现
    '''
    uav_ids = sorted(env.uavs.keys())
    n_agents = len(uav_ids)
    score_history = []

    for i in range(n_eval):
        obs = env.reset()
        score = 0
        terminal = [False] * n_agents
        while not any(terminal):
            # 获取每个智能体的观察
            obs_list = [obs[agent_id] for agent_id in uav_ids]
            # 智能体选择动作
            actions_list = agents.choose_action(obs_list, evaluate=True)
            actions = {agent_id: actions_list[agent_id] for agent_id in uav_ids}
            # print(f"actions:{actions}")
            # 在环境中执行动作
            obs_, rewards, dones, truncs, info = env.step(actions)
            # print(f"rewards:{rewards}")

            # 处理奖励、终止标志和新观察
            rewards_list = [rewards[uav_id] for uav_id in uav_ids]
            dones_list = [dones[uav_id] for uav_id in uav_ids]
            truncs_list = [truncs[uav_id] for uav_id in uav_ids]
            terminal = [dones_list[idx] or truncs_list[idx] for idx in range(n_agents)]

            # print(f"Terminal Flags: {terminal}")

            obs = obs_
            score += sum(rewards_list)
        score_history.append(score)

    avg_score = np.mean(score_history)
    print(f'average score {avg_score}')
    return avg_score


def run():
    env = MultiUAVEnv()  # 初始化多无人机与 IoT 场景
    n_agents = len(env.uavs)  # 无人机数量
    uav_ids = sorted(env.uavs.keys())
    actor_dims = []  # 存储每个智能体的观察空间的维度（从环境中获取）
    n_actions = []  # 存储每个智能体的动作空间的维度

    for agent_id in uav_ids:
        actor_dims.append(env.observation_spaces[agent_id].shape[0])
        n_actions.append(env.action_spaces[agent_id].shape[0])
    critic_dims = sum(actor_dims) + sum(n_actions)

    # 创建 MADDPG 实例

    # 减小学习率
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
                           env=env,
                           gamma=0.99,  # 增大折扣因子
                           alpha=5e-5,  # 减小actor学习率
                           beta=5e-4)  # 减小critic学习率

    critic_dims = sum(actor_dims)

    # 创建一个回放池（Replay Buffer），用于存储智能体的经验并供训练使用
    memory = MultiAgentReplayBuffer(500_000, critic_dims, actor_dims,
                                    n_actions, n_agents, batch_size=256)

    EVAL_INTERVAL = 10  # 每隔 10 步进行评估
    MAX_STEPS = 10000  # 最大步骤数

    total_steps = 0
    episode = 0
    eval_scores = []
    eval_steps = []

    score = evaluate(maddpg_agents, env, episode, total_steps)
    eval_scores.append(score)
    eval_steps.append(total_steps)

    while total_steps < MAX_STEPS:
        observations = env.reset()
        dones = [False] * n_agents
        while not any(dones):
            # 获取每个智能体的观察
            raw_obs = [observations[agent_id] for agent_id in uav_ids]
            actions = maddpg_agents.choose_action(raw_obs)
            next_observations, rewards, dones_dict, truncs, infos = env.step(actions)
            # 处理终止标志
            dones = [dones_dict[agent_id] or truncs[agent_id] for agent_id in uav_ids]

            # 收集奖励
            reward = [rewards[uav_id] for uav_id in uav_ids]  # 列表格式
            # 获取下一步的观察
            next_raw_obs = [next_observations[agent_id] for agent_id in uav_ids]
            # 处理 done 标志
            done = [dones_dict[agent_id] for agent_id in uav_ids]
            state = np.concatenate(raw_obs)
            next_state = np.concatenate(next_raw_obs)

            # 存储经验
            memory.store_transition(raw_obs, state, actions, reward, next_raw_obs, next_state, done)

            # 如果总步骤数是 100 的倍数，执行学习
            if total_steps % 100 == 0 and memory.ready():
                maddpg_agents.learn(memory)
            observations = next_observations
            total_steps += 1

        # 评估
        if total_steps % EVAL_INTERVAL == 0 and episode != 0:
            score = evaluate(maddpg_agents, env, episode, total_steps)
            eval_scores.append(score)
            eval_steps.append(total_steps)

        episode += 1

        # 统计任务成功率
        num_completed_tasks = sum([1 for device in env.devices if device.is_task_completed])
        total_tasks = len(env.devices)
        success_rate = num_completed_tasks / total_tasks if total_tasks > 0 else 1.0
        print(f"Episode {episode} - Task Success Rate: {success_rate * 100:.2f}%")

    # 保存评估结果
    np.save('../data/my_maddpg_scores.npy', np.array(eval_scores))
    np.save('../data/my_maddpg_steps.npy', np.array(eval_steps))


if __name__ == '__main__':
    run()
