'''
本篇内容为DQN 算法的实现
环境是cartpole小车平衡环境
在车杆环境中，有一辆小车，智能体的任务是通过左右移动保持车上的杆竖直
若杆的倾斜度数过大，或者车子离初始位置左右的偏离程度过大，或者坚持时间到达 200 帧，则游戏结束。
状态空间[位置，速度，杆的角度， 杆尖端的速度]
动作空间[左移，右移]
使用神经网络来拟合Q函数
经验回放： 回放缓冲区——从环境中采样一定数量的样本，存储在缓冲区中，然后从缓冲区中随机采样一批样本进行训练
注意，本篇代码的gym版本为0.25.2 过高版本的gym可能会出现问题
'''
import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
#import rl_utils

# 首先定义经验回放池的类，主要包括加入数据、采样数据两大函数
class ReplayBuffer:
    def __init__(self, capacity):
        #collections是Python内建的一个集合模块，提供了许多有用的集合类
        #collections.deque是一个双端队列，可以实现从队列的任意一端快速的插入和删除
        #capacity是缓冲区的大小
        #buffer是一个双端队列，最大长度为capacity
        self.buffer = collections.deque(maxlen=capacity)
    #加入数据的函数
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    #采样数据的函数
    def sample(self, batch_size):
        #batch_size是每次采样的数据量
        #从buffer中采样数据，数量为batch_size。
        #每个数据是一个元组，包括state, action, reward, next_state, done
        transitions = random.sample(self.buffer, batch_size)
        #zip(*)是将数据解压，将state, action, reward, next_state, done分别存储
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done
    #返回buffer中数据的数量
    def size(self):
        return len(self.buffer)
    
#定义一个简单的神经网络，包括一个全连接层和一个输出层
#输入是状态，输出是动作的Q值
#只有一层隐藏层的Q 网络
class Qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        #super()里参数是子类名和self
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        #传递agent的状态x进入神经Q网络，维度为state_dim
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device):
        #初始化参数
        self.action_dim = action_dim
        #初始化Q网络, .to(device)是将网络放到GPU上
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        #初始化目标Q网络
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        #使用Adam优化器,更新的是Q网络的参数，不是目标Q网络的参数
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma #折扣因子
        self.epsilon = epsilon #epsilon-greedy策略的epsilon
        self.target_update = target_update #目标网络的更新频率
        self.count = 0 #计数器，记录更新次数
        self.device = device #gpu或cpu
    
    def take_action(self, state):
        #epsilon-greedy策略 采取动作
        #如果随机数小于epsilon，随机选择一个动作
        #self.epsilon是一个超参数，用来控制探索的程度，是人为设定的
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            #否则选择Q值最大的动作
            #state是agent的状态，维度为state_dim, 从列表转换为tensor 张量
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action
    def update(self, transition_dict):
        #在环境里更新Q网络
        #把所有的东西以字典的形式存储在transition_dict的字典里
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        #注意，这里要使用view函数把张量的维度转换为1列张量
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],dtype = torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)

        #计算Q值, gather函数是根据索引值取出对应的元素
        q_values = self.q_net(states).gather(1, actions)
        #计算下一个状态的最大的Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1,1)
        #依据td误差目标函数计算Q值
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        #均方误差计算损失函数
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        #pytorch中默认梯度会进行累积，这里需要显示的将梯度设置为0
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()
        #每隔一定的步数更新目标网络
        if self.count % self.target_update == 0:
            #将Q网络的参数复制到目标Q网络
            #state_dict()函数是将神经网络的参数以字典的形式返回
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

#一切准备就绪，现在准备训练
if __name__ == '__main__':
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    # 这里设置四个随机种子，
    # 一个是python内置函数random，一个是numpy的random，
    # 一个是环境的seed，一个是torch的seed
    # 四者不同，本篇代码中都用到了，所以都要设置
    random.seed(0)
    np.random.seed(0) 
    env.seed(0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)
    #env.observation_space是gym环境的状态空间，.shape[0]是状态空间的维度
    state_dim = env.observation_space.shape[0]
    #env.action_space.n是动作空间的维度, 需要根据具体的环境来设置
    action_dim = env.action_space.n
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
    #return_list是用来存储每一轮的回报
    return_list = []
    for i in range(10):
        with tqdm(total = int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset() #每一轮episode开始时，重置环境
                done = False
                while not done:
                    action = agent.take_action(state)
                    #执行动作，得到下一个状态，奖励，是否结束, _是额外信息
                    next_state, reward, done, _ = env.step(action)
                    #将数据加入到经验回放池中
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    #当经验回放池中的数据大于最小值时，超过一定值时后，开始训练
                    if replay_buffer.size() > minimal_size:
                        #从经验回放池中采样数据
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'rewards': b_r,
                            'next_states': b_ns,
                            'dones': b_d
                        }
                        #更新Q网络
                        agent.update(transition_dict)
                #当每一轮的episode以done结束时，将总回报加入到return_list中
                return_list.append(episode_return)
                #每隔10次episode打印一次回报
                if (i_episode + 1) % 10 == 0 :
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode +1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
            

#接下来画一些回报图
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Return')
plt.title('DQN on {}'.format(env_name))
plt.savefig('DQN_{}.png'.format(env_name))

















