import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CriticNetwork(nn.Module):
    '''
    Critic 网络是基于当前状态和动作来估计 Q-value，目的是告诉 Actor 网络每个动作的“好坏”
    '''
    def __init__(self, beta, input_dims, fc1, fc2,
                 name, chkpt_dir):
        '''
        :param beta:学习率, 控制 Critic 网络在训练时权重更新的步长
        :param input_dims:输入维度，通常是状态空间的维度和动作空间的维度之和
        :param fc1:全连接层的隐藏单元数
        :param fc2:全连接层的隐藏单元数
        :param name:模型的名称，用于标识和保存模型的检查点
        :param chkpt_dir:检查点保存的目录路径，保存训练的模型参数。
        '''
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims  # 存储输入维度

        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.fc1 = nn.Linear(input_dims, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.q = nn.Linear(fc2, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        '''
        state 和 action 被拼接在一起，然后通过 fc1 和 fc2 层。
        经过激活函数（ReLU）后，最终通过输出层 q 计算 Q-value
        '''
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        # 保存模型的参数
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        # 加载已保存的模型参数
        self.load_state_dict(T.load(self.chkpt_file))


class ActorNetwork(nn.Module):
    '''
    ActorNetwork 用于生成策略，即根据给定的状态输出一个具体的动作。决定了智能体在某一时刻采取的动作。
    '''
    def __init__(self, alpha, input_dims, fc1, fc2,
                 n_actions, name, chkpt_dir):
        '''
        :param alpha:
        :param input_dims:输入的状态维度。
        :param fc1:全连接层的隐藏单元数
        :param fc2:全连接层的隐藏单元数
        :param n_actions: 动作空间的维度
        :param name:
        :param chkpt_dir:检查点文件保存目录
        '''
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims  # 存储输入维度

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        # pi 表示策略网络的输出层，它的输出是每个动作的概率分布。
        self.pi = nn.Linear(fc2, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        '''
        对输入状态应用两层全连接层，并使用 ReLU 激活函数
        最后通过 pi 层生成动作 pi，并使用 T.tanh 将其限制在范围 [-1, 1]。
        '''
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = T.tanh(self.pi(x))

        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))
