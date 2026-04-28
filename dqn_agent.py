import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim=256):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, n_actions, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=5000,
                 buffer_capacity=20000, batch_size=64, target_update=100,
                 device='cpu'):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start          # <-- THÊM DÒNG NÀY
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_cnt = 0
        self.device = device

        self.policy_net = DQN(state_dim, n_actions).to(device)
        self.target_net = DQN(state_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_capacity)

    def select_action(self, state, evaluate=False):
        if not evaluate and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return q_values.argmax().item()

    def update_epsilon(self, step):
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       np.exp(-1. * step / self.epsilon_decay)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None, None
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        state_t = torch.FloatTensor(state).to(self.device)
        action_t = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward_t = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state_t = torch.FloatTensor(next_state).to(self.device)
        done_t = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        q_values = self.policy_net(state_t).gather(1, action_t)

        with torch.no_grad():
            next_q = self.target_net(next_state_t).max(1, keepdim=True)[0]
            target_q = reward_t + self.gamma * next_q * (1 - done_t)

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_cnt += 1
        if self.update_cnt % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item(), q_values.mean().item()