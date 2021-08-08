import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import gym
import gym, threading, queue
import numpy as np
EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 3, 1

METHOD = [
    {"name":'kl_pen', "kl_target":0.01, "lam":0.5},   # KL penalty
    {"name":'clip', "epsilon":0.2}   # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization
class Critic(nn.Module):
    def __init__(self,s_dim):
        super(Critic, self).__init__()
        self.s_dim = s_dim
        self.fc1 = nn.Linear(s_dim,100)
        self.relu = nn.ReLU()
        self.v = nn.Linear(100,1)
    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.v(x)
        return x
    def loss_func(self,s,tfdc_r):
        v = self.forward(s)
        advantage = tfdc_r - v
        loss = torch.mean(advantage**2)
        return loss
class Actor(nn.Module):
    def __init__(self,s_dim,a_dim):
        super(Actor, self).__init__()
        self.s_dim = s_dim
        self.fc1 = nn.Linear(s_dim,100)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()
        self.fc_mu = nn.Linear(100,a_dim)
        self.fc_sigma = nn.Linear(100,a_dim)
    def forward(self,x):
        x = self.relu(self.fc1(x))
        mu = 2*self.tanh(self.fc_mu(x)).squeeze(0)
        sigma = self.softplus(self.fc_sigma(x)).squeeze(0)
        # print(mu,sigma)
        return mu,sigma

class PPO(object):
    def __init__(self):
        self.critic = Critic(S_DIM)
        self.pi = Actor(S_DIM,A_DIM)
        self.opt_a = torch.optim.Adam(self.pi.parameters(),lr=A_LR)
        self.opt_c = torch.optim.Adam(self.critic.parameters(),lr=C_LR)
    def update(self, s, a, r):
        if s.ndim < 2: s = s[np.newaxis, :]
        s = torch.tensor(s,dtype=torch.float)
        r = torch.tensor(r,dtype=torch.float)
        a = torch.tensor(a,dtype=torch.float)
        mu, sigma = self.pi(s)
        old_dist = torch.distributions.Normal(mu.detach().clone(),sigma.detach().clone())
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful
        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                v = self.critic(s)
                adv = r - v
                mu, sigma = self.pi(s)
                dist = torch.distributions.Normal(mu,sigma)
                ratio = torch.exp(dist.log_prob(a)) / (torch.exp(old_dist.log_prob(a)) + 1e-5)
                surr = ratio * adv
                kl = torch.distributions.kl.kl_divergence(old_dist, dist)
                kl_mean = torch.mean(kl)
                aloss = - torch.mean(surr - METHOD['lam']* kl)
                self.opt_a.zero_grad()
                aloss.backward()
                self.opt_a.step()
                if kl_mean > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl_mean < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl_mean > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            for _ in range(A_UPDATE_STEPS):
                v = self.critic(s)
                adv = r - v
                mu, sigma = self.pi(s)
                dist = torch.distributions.Normal(mu,sigma)
                ratio = torch.exp(dist.log_prob(a)) / (torch.exp(old_dist.log_prob(a)) + 1e-5)
                surr = ratio * adv
                aloss = -torch.mean(torch.min(
                    surr,
                    torch.clamp(ratio,1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*adv
                ))
                self.opt_a.zero_grad()
                aloss.backward()
                self.opt_a.step()
        # update critic
        for _ in range(C_UPDATE_STEPS):
            self.opt_c.zero_grad()
            closs = self.critic.loss_func(s,r)
            closs.backward()
            self.opt_c.step()        
    def choose_action(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        s = torch.tensor(s, dtype = torch.float)
        mu,sigma = self.pi(s)
        dist = torch.distributions.Normal(mu,sigma)
        a = dist.sample([1])[0].detach().numpy()
        return np.clip(a, -2, 2)
    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        s = torch.tensor(s,dtype=torch.float)
        return self.critic(s).detach().numpy()[0, 0]
        

env = gym.make('Pendulum-v0').unwrapped
ppo = PPO()
all_ep_r = []

for ep in range(EP_MAX):
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    for t in range(EP_LEN):    # in one episode
        # env.render()
        a = ppo.choose_action(s)
        s_, r, done, _ = env.step(a)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8)    # normalize reward, find to be useful
        s = s_
        ep_r += r

        # update ppo
        if (t+1) % BATCH == 0 or t == EP_LEN-1:
            v_s_ = ppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(bs, ba, br)
    if ep == 0: all_ep_r.append(ep_r)
    else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
    print(
        'Ep: %i' % ep,
        "|Ep_r: %i" % ep_r,
        ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
    )

plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()