import numpy as np
import torch as T
import torch.nn.functional as F
from networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions,
                 n_agents, agent_idx, chkpt_dir, min_action,
                 max_action, alpha=1e-4, beta=1e-3, fc1=64,
                 fc2=64, gamma=0.95, tau=0.01):
        '''
        :param actor_dims:Actor 网络的输入维度（通常是状态的维度）
        :param critic_dims:Critic 网络的输入维度（通常是状态和动作的联合维度）
        :param n_actions:动作空间的维度
        :param n_agents:环境中的智能体数量
        :param agent_idx:当前智能体的索引，区分不同的智能体
        :param chkpt_dir:模型检查点的保存目录
        :param min_action:动作的最小值
        :param max_action:最大值
        :param alpha:学习率 Actor
        :param beta:学习率 Critic
        :param fc1:网络隐藏层的神经元数
        :param fc2:网络隐藏层的神经元数
        :param gamma:折扣因子，表示未来奖励的影响程度
        :param tau:目标网络的更新软更新因子，用于平滑更新目标网络
        '''
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        agent_name = 'agent_%s' % agent_idx
        self.agent_idx = agent_idx
        self.min_action = min_action
        self.max_action = max_action

        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                  chkpt_dir=chkpt_dir,
                                  name=agent_name+'_actor')

        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2,
                                         n_actions, chkpt_dir=chkpt_dir,
                                         name=agent_name+'target__actor')

        self.critic = CriticNetwork(beta, critic_dims, fc1, fc2,
                                    chkpt_dir=chkpt_dir,
                                    name=agent_name+'_critic')
        self.target_critic = CriticNetwork(beta, critic_dims, fc1, fc2,
                                           chkpt_dir=chkpt_dir,
                                           name=agent_name+'_target__critic')

        self.update_network_parameters(tau=1)


    def choose_action(self, observation, evaluate=False):
        '''
        根据当前状态生成动作
        '''
        state = T.tensor(observation[np.newaxis, :], dtype=T.float, device=self.actor.device)
        actions = self.actor.forward(state)

        noise = T.randn(size=(self.n_actions,)).to(self.actor.device)
        noise *= T.tensor(1 - int(evaluate))
        action = T.clamp(actions + noise,
                         T.tensor(self.min_action, device=self.actor.device),
                         T.tensor(self.max_action, device=self.actor.device))
        return action.data.cpu().numpy()[0]

    def update_network_parameters(self, tau=None):
        '''
        软更新（Soft Update），将 Actor 和 Critic 网络的目标网络更新为当前网络的加权平均
        '''
        tau = tau or self.tau

        src = self.actor
        dest = self.target_actor

        for param, target in zip(src.parameters(), dest.parameters()):
            target.data.copy_(tau * param.data + (1 - tau) * target.data)

        src = self.critic
        dest = self.target_critic

        for param, target in zip(src.parameters(), dest.parameters()):
            target.data.copy_(tau * param.data + (1 - tau) * target.data)

    def save_models(self):
        # 保存 Actor 和 Critic 网络的模型参数，使用之前定义的 save_checkpoint() 方法
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        # 加载保存的模型参数，恢复之前训练过的网络
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self, memory, agent_list):
        '''
        从经验回放池（memory）中抽取样本，通过 Q-learning（强化学习中的价值迭代）优化 Actor 和 Critic 网络的参数
        :param memory: 经验回放池
        :param agent_list:包含所有智能体的列表.在 MADDPG 算法中，每个智能体在更新时需要考虑（访问）其他智能体的动作
        Critic 网络 通过计算当前状态和动作的 Q 值，来估计未来的奖励。
        Actor 网络 通过最大化 Critic 网络计算的 Q 值，来优化其策略。
        使用 经验回放池 来打破数据的时间相关性，增强学习的稳定性和效率。
        '''

        # 检查经验回放池是否有足够的数据 如果不足，直接返回
        if not memory.ready():
            return

        # 从经验回放池中采样数据 状态 环境 动作 奖励 下一状态 下一环境 该经验是否结束（终止标志）
        actor_states, states, actions, rewards, actor_new_states, states_, dones = memory.sample_buffer()

        # 将每个智能体的状态、下一状态和动作转换为 PyTorch 张量
        device = self.actor.device

        states = T.tensor(np.array(states), dtype=T.float, device=device)
        rewards = T.tensor(np.array(rewards), dtype=T.float, device=device)
        states_ = T.tensor(np.array(states_), dtype=T.float, device=device)
        dones = T.tensor(np.array(dones), device=device)

        actor_states = [T.tensor(actor_states[idx], device=device, dtype=T.float) for idx in range(len(agent_list))]
        actor_new_states = [T.tensor(actor_new_states[idx], device=device, dtype=T.float) for idx in range(len(agent_list))]
        actions = [T.tensor(actions[idx], device=device, dtype=T.float) for idx in range(len(agent_list))]

        # 计算目标 Q 值
        with T.no_grad():
            new_actions = T.cat([agent.target_actor(actor_new_states[idx]) for idx, agent in enumerate(agent_list)], dim=1)
            critic_value_ = self.target_critic.forward(states_, new_actions).squeeze()
            critic_value_[dones[:, self.agent_idx]] = 0.0
            target = rewards[:, self.agent_idx] + self.gamma * critic_value_

        # 计算 Critic 网络损失
        old_actions = T.cat([actions[idx] for idx in range(len(agent_list))], dim=1)
        critic_value = self.critic.forward(states, old_actions).squeeze()
        critic_loss = F.mse_loss(target, critic_value)

        # 更新 Critic 网络
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic.optimizer.step()

        # 计算 Actor 网络损失
        actions[self.agent_idx] = self.actor.forward(actor_states[self.agent_idx])
        actions = T.cat([actions[i] for i in range(len(agent_list))], dim=1)
        actor_loss = -self.critic.forward(states, actions).mean()

        # 更新 Actor 网络
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor.optimizer.step()

        # 更新目标网络（软更新）
        self.update_network_parameters()
