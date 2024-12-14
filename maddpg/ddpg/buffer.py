import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = []
        self.new_state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.terminal_memory = []

    def store_transition(self, state, action, reward, state_, done):
        if self.mem_cntr < self.mem_size:
            # 如果缓冲区未满，使用 append 方法
            self.state_memory.append(state)
            self.new_state_memory.append(state_)
            self.action_memory.append(action)
            self.reward_memory.append(reward)
            self.terminal_memory.append(done)
        else:
            # 如果缓冲区已满，覆盖最旧的数据
            index = self.mem_cntr % self.mem_size
            self.state_memory[index] = state
            self.new_state_memory[index] = state_
            self.action_memory[index] = action
            self.reward_memory[index] = reward
            self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = np.array([self.state_memory[idx] for idx in batch])  # 修改为列表推导
        actions = np.array([self.action_memory[idx] for idx in batch])
        rewards = np.array([self.reward_memory[idx] for idx in batch])
        states_ = np.array([self.new_state_memory[idx] for idx in batch])
        terminal = np.array([self.terminal_memory[idx] for idx in batch])

        return states, actions, rewards, states_, terminal



