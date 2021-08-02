"""
The Dueling DQN based on this paper: https://arxiv.org/abs/1511.06581

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import torch
import torch.nn as nn

np.random.seed(1)
class Net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS,dueling,n_l1=100):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, n_l1)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.relu = nn.ReLU()
        self.dueling = dueling
        if self.dueling:
            self.fc_state = nn.Linear(n_l1,1)
            self.fc_state.weight.data.normal_(0, 0.1)   # initialization
            self.fc_action = nn.Linear(n_l1,N_ACTIONS)
            self.fc_action.weight.data.normal_(0, 0.1)   # initialization
        else:
            self.out = nn.Linear(n_l1, N_ACTIONS)
            self.out.weight.data.normal_(0, 0.1)   # initialization
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        if self.dueling:
            s = self.fc_state(x)
            a = self.fc_action(x)
            actions_value = s + a - torch.mean(a,dim = 1).unsqueeze(1)
        else:
            actions_value = self.out(x)
        return actions_value

class DuelingDQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            dueling=True,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.dueling = dueling      # decide to use dueling DQN or not
        self.eval_net = Net(self.n_features,self.n_actions,self.dueling)
        self.target_net = Net(self.n_features,self.n_actions,self.dueling)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.loss_func = nn.MSELoss()
        self.opt = torch.optim.RMSprop(self.eval_net.parameters(),lr=learning_rate)
        
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features*2+2))
        
        self.loss_his = []

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = torch.tensor(observation[np.newaxis, :],dtype=torch.float)
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.eval_net(observation).detach().numpy()
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        eval_act_index = torch.tensor(batch_memory[:, self.n_features:self.n_features+1],dtype=torch.long)
        q_eval = self.eval_net(torch.tensor(batch_memory[:, :self.n_features],dtype=torch.float)).gather(1,eval_act_index)
        q_next = self.target_net(torch.tensor(batch_memory[:, -self.n_features:],dtype=torch.float)).detach()
        reward = torch.tensor(batch_memory[:, self.n_features + 1:self.n_features + 2],dtype=torch.float)
        q_target = reward + self.gamma*q_next.max(1)[0].unsqueeze(1)
        loss = self.loss_func(q_eval,q_target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.loss_his.append(loss.item())

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1





