import numpy as np
#origin

class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims,
                 n_actions, n_agents, batch_size):
        '''
        :param max_size:回放池的最大容量
        :param critic_dims:Critic 网络的输入维度
        :param actor_dims:每个智能体的 Actor 网络 输入维度
        :param n_actions:每个智能体的动作空间维度
        :param n_agents:智能体的数量
        :param batch_size:从回放池中每次采样的经验数量
        '''
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions

        # 内存数组，分别存储每个智能体的经验数据
        self.state_memory = np.zeros((self.mem_size, critic_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, critic_dims), dtype=np.float32)
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

        # 初始化了每个智能体的 Actor 相关的内存
        self.init_actor_memory()

    def init_actor_memory(self):
        '''
         初始化智能体的 Actor 相关内存
        存储每个智能体在每个时间步的状态 下一状态 选择的动作
        '''
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_memory.append(np.zeros((self.mem_size, self.n_actions[i])))

    def store_transition(self, raw_obs, state, action, reward,
                         raw_obs_, state_, done):
        '''
        将当前智能体与环境交互的经验存储到回放池中
        将当前的 观察、状态、动作、奖励、下一状态、终止标志 存储到相应的内存中
        '''

        index = self.mem_cntr % self.mem_size    # 取余 确保在存储时不会超出回放池的最大容量
        # 遍历所有智能体，将它们的经验存储到相应的内存数组
        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[f'uav_{agent_idx}']

        # 接着将环境状态、奖励、终止标志等存储到回放池的对应位置
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self):
        '''
        从回放池中采样
        :return:
        '''
        max_mem = min(self.mem_cntr, self.mem_size)

        # 从回放池中随机选择一个批次，接着将采样到的批次数据提取出来
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        actor_states = []
        actor_new_states = []
        actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(
                self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])

        return actor_states, states, actions, rewards, actor_new_states, states_, terminal

    def ready(self):
        '''
        检查回放池中是否有足够的经验供训练使用
        '''
        if self.mem_cntr >= self.batch_size:
            return True
        return False
