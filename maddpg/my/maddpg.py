from agent import Agent
import numpy as np

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, env,
                 alpha=1e-4, beta=1e-3, fc1=64, fc2=64, gamma=0.95, tau=0.01,
                 chkpt_dir='tmp/maddpg/', scenario='co-op_navigation'):
        # scenario: 场景名称，用于为不同场景保存和加载模型
        # 创建智能体列表：
        self.agents = []   # 用于存储所有智能体（Agent）对象的列表
        chkpt_dir += scenario        #将 scenario 添加到 chkpt_dir 以区分不同场景的模型保存路径
        for agent_idx in range(n_agents):      #遍历每个智能体，创建相应的 Agent 对象
            agent = list(env.action_spaces.keys())[agent_idx]    #获取每个智能体的名字或 ID
            min_action = env.action_space[agent].low     #获取每个智能体动作空间的最小值和最大值
            max_action = env.action_space[agent].high
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,
                               n_actions[agent_idx], n_agents, agent_idx,
                               alpha=alpha, beta=beta, tau=tau, fc1=fc1,
                               fc2=fc2, chkpt_dir=chkpt_dir,
                               gamma=gamma, min_action=min_action,
                               max_action=max_action))


    def save_checkpoint(self):
        '''
         保存所有智能体的模型
         调用每个智能体的 save_models 方法，保存 Actor 和 Critic 网络的参数
        '''
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        '''
        加载所有智能体的模型 加载 Actor 和 Critic 网络的参数
        '''
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs, evaluate=False):
        '''
        根据当前状态选择智能体的动作
        返回包含所有智能体动作的字典 actions
        '''
        actions = {}
        for idx, agent in enumerate(self.agents):  # 用索引而不是直接解包 raw_obs
            observation = raw_obs[idx]  # 获取对应智能体的观测
            action = agent.choose_action(observation, evaluate)  # 调用 Agent.choose_action
            action = np.clip(action, 0.0, 1.0)
            actions[agent_id] = action
            actions[f'uav_{idx}'] = action  # 用智能体标识符作为键
        return actions

    def learn(self, memory):
        '''
        用于进行智能体的训练，增加协作和调试信息
        '''
        for agent_idx, agent in enumerate(self.agents):
            # 为每个智能体提供其他智能体的动作信息
            agent.learn(memory, self.agents)

