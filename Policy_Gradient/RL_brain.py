"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import torch
import torch.nn as nn
# reproducible
np.random.seed(1)

def one_hot(acts,n_actions):
    batch_size = len(acts)
    length = n_actions
    y_onehot = torch.zeros([batch_size, length])
    return y_onehot.scatter_(1, acts, 1)
class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.net = self.create_net()
        self.opt = torch.optim.Adam(self.net.parameters(),lr=learning_rate)
        self.loss_his = []
        
        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

    def init_weights(self,m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, 0.1)
    def create_net(self):
        model = nn.Sequential(
              nn.Linear(self.n_features,30),
              nn.Tanh(),
              nn.Linear(30,self.n_actions),
              nn.Softmax(dim=1)
            )
        model=model.apply(self.init_weights)
        return model
    
        
    def choose_action(self, observation):
        observation = torch.tensor(observation[np.newaxis, :],dtype=torch.float)
        prob_weights = self.net(observation).detach().numpy()
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        obs = torch.tensor(np.vstack(self.ep_obs),dtype=torch.float)
        actions = torch.tensor(np.array(self.ep_as),dtype=torch.long)
        vt = torch.tensor(discounted_ep_rs_norm,dtype=torch.float)
        # train on episode
        all_act_prob = self.net(obs)
        loss = torch.sum(-torch.log(all_act_prob)*one_hot(actions.unsqueeze(1), self.n_actions)*vt)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs



