"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""
import torch
import torch.nn as nn
import numpy as np
import gym
import time


#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.001      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
ENV_NAME = 'Pendulum-v0'
###############################  DDPG  ####################################
class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, n_l1=30):
        super(Actor,self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(s_dim,n_l1)
        self.fc2 = nn.Linear(n_l1,a_dim)
    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return x
def smooth_replace(eval_net,target_net):
    eval_dict = eval_net.state_dict()
    # print(eval_dict)
    target_dict = target_net.state_dict()
    new_dict = dict()
    for (k,v) in target_dict.items():
        new_dict[k] = TAU*eval_dict[k] + (1-TAU)*target_dict[k]
    target_net.load_state_dict(new_dict)
class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, n_l1=30):
        super(Critic, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.fc_s = nn.Linear(s_dim,n_l1)
        self.fc_a = nn.Linear(a_dim,n_l1)
        self.relu = nn.ReLU()
        self.out = nn.Linear(n_l1,1)
    def forward(self,s,a):
        s = self.fc_s(s)
        a = self.fc_a(a)
        x = s+a
        x = self.relu(x)
        x = self.out(x)
        return x


class DDPG(object):
    def __init__(self, a_dim, s_dim):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        
        self.a_dim, self.s_dim, = a_dim, s_dim,
        # actor
        self.a = Actor(s_dim,a_dim)
        self.a_ = Actor(s_dim,a_dim)
        self.q = Critic(s_dim,a_dim)
        self.q_ = Critic(s_dim,a_dim)
        
        self.loss_func = nn.MSELoss()
        self.opt_q = torch.optim.Adam(self.q.parameters(),lr=LR_C)
        self.opt_a = torch.optim.Adam(self.a.parameters(),lr=LR_A)
        # q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        # td_error = tf.compat.v1.losses.mean_squared_error(labels=q_target, predictions=q)
        # self.ctrain = tf.compat.v1.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        # a_loss = - tf.reduce_mean(input_tensor=q)    # maximize the q
        # self.atrain = tf.compat.v1.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)
    def choose_action(self, s):
        s = torch.tensor(s[np.newaxis, :],dtype=torch.float)
        action = self.a(s)[0].detach().numpy()
        # print(action)
        return action

    def learn(self):
        # torch.autograd.set_detect_anomaly(True)
        # soft target replacement
        smooth_replace(self.a,self.a_)
        smooth_replace(self.q,self.q_)
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = torch.tensor(bt[:, :self.s_dim],dtype=torch.float)
        ba = torch.tensor(bt[:, self.s_dim: self.s_dim + self.a_dim],dtype=torch.float)
        br = torch.tensor(bt[:, -self.s_dim - 1: -self.s_dim],dtype=torch.float)
        bs_ = torch.tensor(bt[:, -self.s_dim:],dtype=torch.float)
        self.opt_a.zero_grad()
        q =  self.q(bs,self.a(bs))
        loss = - q.mean()
        loss.backward()
        self.opt_a.step()
        self.opt_q.zero_grad()
        q =  self.q(bs,ba)
        q_target = br + GAMMA * self.q_(bs_,self.a_(bs_))
        td_error = self.loss_func(q,q_target)
        td_error.backward()
        self.opt_q.step()
        
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

###############################  training  ####################################

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
print(a_dim)
a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim)

var = 3  # control exploration
t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER and i>150:
            env.render()

        # Add exploration noise
        a = ddpg.choose_action(s)*a_bound
        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a/a_bound, r / 10, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            ddpg.learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            # if ep_reward > -300:RENDER = True
            break
print('Running time: ', time.time() - t1)