"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.

The cart pole example. Policy is oscillated.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import numpy as np
import torch
import gym
import torch.nn as nn

np.random.seed(2)

# Superparameters
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = True  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n


class Actor_Net(nn.Module):
    def __init__(self,n_features,n_actions,n_l1=20):
        super(Actor_Net, self).__init__()
        self.fc1 = nn.Linear(n_features,n_l1)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc1.bias.data.fill_(0.1)
        self.fc2 = nn.Linear(n_l1,n_actions)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc2.bias.data.fill_(0.1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))        
        return x

class Actor(object):
    def __init__(self,n_features, n_actions, lr=0.0001):
        # s,a,td_error
        self.net = Actor_Net(n_features,n_actions)
        # self.e = epsilon = tf.train.exponential_decay(2., global_step, 1000, 0.9)
        self.normal_dist = torch.distributions.Normal(0,1)
        self.opt = torch.optim.Adam(self.net.parameters(),lr=lr)
        self.loss_func = nn.MSELoss()
        self.action = None
    
    def learn(self, td_error):
        td_error = td_error.clone().detach()
        log_prob = torch.log(self.probs)
        self.actions = torch.tensor(self.actions,dtype=torch.long)
        log_prob = log_prob[0,self.actions]
        exp_v = torch.mean(log_prob*td_error)
        loss = -exp_v
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
    def choose_action(self, s):
        s = torch.tensor(s[np.newaxis, :],dtype=torch.float)
        probs = self.net(s)
        self.probs = probs
        self.actions = np.random.choice(np.arange(probs.shape[1]), p=probs.detach().numpy().ravel())
        return self.actions

class Critic_Net(nn.Module):
    def __init__(self,n_features,n_l1=20):
        super(Critic_Net, self).__init__()
        self.fc1 = nn.Linear(n_features,n_l1)
        self.relu = nn.ReLU()
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc1.bias.data.fill_(0.1)
        self.fc2 = nn.Linear(n_l1,1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc2.bias.data.fill_(0.1)
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
        
        
class Critic(object):
    def __init__(self, n_features, lr=0.01):
        self.net = Critic_Net(n_features)
        self.opt = torch.optim.Adam(self.net.parameters(),lr=lr)
    def learn(self, s, r, s_):
        s = torch.tensor(s[np.newaxis, :],dtype=torch.float)
        s_ = torch.tensor(s_[np.newaxis,:],dtype=torch.float)
        r = torch.tensor(r,dtype=torch.float)
        v =  self.net(s)
        v_ = self.net(s_)
        td_error = torch.mean(r + GAMMA*v_ - v)
        loss = td_error**2
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return td_error

actor = Actor( n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic( n_features=N_F, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor


for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER and i_episode>1000: env.render()

        a = actor.choose_action(s)

        s_, r, done, info = env.step(a)

        if done: r = -20

        track_r.append(r)

        td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break

