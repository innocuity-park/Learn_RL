'''
本篇代码为ppo算法的实现，主要为ppo-截断
'''
import gym
import torch
import torch.nn.functional as F 
import numpy as np
import matplotlib.pyplot as plt
import rl_utils

class PolicyNet(torch.nn.Module):
    '''
    定义强化学习里的策略网络
    策略网络用于输出给定状态下的动作概率分布。
    它的目标是学习一个策略，使得在给定状态下选择的动作能够最大化累积奖励。
    '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)

class ValueNet(torch.nn.Module):
    '''
    定义强化学习里的价值网络
    价值网络用于评估给定状态的价值。
    价值网络的目标是学习一个价值函数，使得在给定状态下的价值能够最大化累积奖励。
    '''
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class PPO:
    '''
    PPO算法，采取截断的方式
    
    '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr,
                 critic_lr, lmbda, epochs, eps, gamma, device):
        #首先定义actor和critic
        #Actor：生成动作的概率分布，并根据这些概率选择动作。
        #Critic：评估当前状态的价值，计算优势函数
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda 
        self.epochs = epochs #一条序列的数据用来训练轮数
        self.eps = eps #ppo中截断范围的参数
        self.device = device
    
    def take_action(self, state):
        state = torch.tensor([state], dtpye=torch.float).to(self.device)
        probs = self.actor(state) #输出动作概率
        #torch.distributions.Categorical(probs)表示一个多项分布，probs是每个动作的概率
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        #计算td_targets，讲下一个状态输入到价值网络中，得到下一个状态的价值
        #再与当前状态的价值相减，得到td误差
        td_targets = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_targets - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        #.detach()是将张量从计算图中分离出来，不再计算梯度
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            #计算当前动作的概率与之前动作的概率的比值
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()





















