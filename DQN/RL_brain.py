import torch
import numpy as np
import pandas as pd
import torch.nn as nn
np.random.seed(1)
# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
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
        # total learning step
        self.learn_step_counter = 0
        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        # consist of [target_net, evaluate_net]
        self.eval_net = self.create_net()
        self.target_net = self.create_net()
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.loss_func = nn.MSELoss()
        self.opt = torch.optim.RMSprop(self.eval_net.parameters(),lr=learning_rate)
        self.loss_his = []

    def init_weights(self,m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, 0.1)
    def create_net(self):
        model = nn.Sequential(
              nn.Linear(self.n_features,50),
              nn.ReLU(),
              nn.Linear(50,self.n_actions)
            )
        model=model.apply(self.init_weights)
        return model
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1
    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
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
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.loss_his)), self.loss_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
       



