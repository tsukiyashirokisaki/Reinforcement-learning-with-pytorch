"""
Actor-Critic with continuous action using TD-error as the Advantage, Reinforcement Learning.

The Pendulum example (based on https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/Continuous%20MountainCar%20Actor%20Critic%20Solution.ipynb)

Cannot converge!!! oscillate!!!

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow r1.3
gym 0.8.0
"""
import torch
import numpy as np
import gym
import torch.nn as nn

np.random.seed(2)
class Actor_Net(nn.Module):
    def __init__(self,n_features,n_l1=30):
        super(Actor_Net, self).__init__()
        self.fc1 = nn.Linear(n_features,n_l1)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc1.bias.data.fill_(0.1)
        self.fc_mu = nn.Linear(n_l1,1)
        self.fc_mu.weight.data.normal_(0, 0.1)
        self.fc_mu.bias.data.fill_(0.1)        
        self.fc_sigma = nn.Linear(n_l1,1)
        self.fc_sigma.weight.data.normal_(0, 0.1)
        self.fc_sigma.bias.data.fill_(1.) 
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.relu(self.fc1(x))
        mu = self.tanh(self.fc_mu(x))
        sigma = self.softplus(self.fc_sigma(x))
        return mu,sigma

class Actor(object):
    def __init__(self, n_features, action_bound, lr=0.0001):
        # s,a,td_error
        self.net = Actor_Net(n_features)
        # self.e = epsilon = tf.train.exponential_decay(2., global_step, 1000, 0.9)
        self.normal_dist = torch.distributions.Normal(0,1)
        self.opt = torch.optim.Adam(self.net.parameters(),lr=lr)
        self.loss_func = nn.MSELoss()
        self.action = None
        self.action_bound = action_bound
    
    def learn(self, td_error):
        td_error = td_error.clone().detach()
        log_prob = self.normal_dist.log_prob(self.action)  # loss without advantage
        exp_v = log_prob * td_error + 0.01*self.normal_dist.entropy()# Add cross entropy cost to encourage exploration
        loss = -exp_v
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return exp_v

    def choose_action(self, s):
        s = torch.tensor(s[np.newaxis, :],dtype=torch.float)
        mu, sigma = self.net(s)
        mu = torch.squeeze(mu*2)
        sigma = torch.squeeze(sigma+0.1)
        self.normal_dist = torch.distributions.Normal(mu,sigma)
        self.action = torch.clamp(self.normal_dist.sample(),float(self.action_bound[0]),float(self.action_bound[1]))
        return self.action.item()

class Critic_Net(nn.Module):
    def __init__(self,n_features,n_l1=30):
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
        td_error = r + GAMMA*v_ - v
        loss = torch.mean(td_error**2)  
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return td_error


MAX_EPISODE = 1000
MAX_EP_STEPS = 200
DISPLAY_REWARD_THRESHOLD = -100  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time
GAMMA = 0.9
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001     # learning rate for critic

env = gym.make('Pendulum-v0')
env.seed(1)  # reproducible
env = env.unwrapped

N_S = env.observation_space.shape[0]
A_BOUND = env.action_space.high


actor = Actor(n_features=N_S, lr=LR_A, action_bound=[-A_BOUND, A_BOUND])
critic = Critic(n_features=N_S, lr=LR_C)


for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    ep_rs = []
    while True:
        # if RENDER:
        # env.render()
        a = actor.choose_action(s)

        s_, r, done, info = env.step(np.array([a]))
        r /= 10

        td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(td_error)  # true_gradient = grad[logPi(s,a) * td_error]

        s = s_
        t += 1
        ep_rs.append(r)
        if t > MAX_EP_STEPS:
            ep_rs_sum = sum(ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.9 + ep_rs_sum * 0.1
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break

