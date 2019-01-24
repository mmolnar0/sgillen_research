
# coding: utf-8

# In[7]:


import torch
import torch.nn as nn
#from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.functional import softmax


# for the training only
from itertools import count
import gym
import torch.optim as optim
import numpy as np



eps = np.finfo(np.float32).eps.item()


class Router(nn.Module):
    def __init__(self, input_size, hidden_size, router_size, output_size):
        super().__init__()
        
        # Routing layer gates
        self.r_linear1 = nn.Linear(input_size, router_size)
        self.r_linear2 = nn.Linear(router_size, 2)
        
        # Swingup layer gates
        self.s_linear1 = nn.Linear(input_size, hidden_size)
        self.s_linear2 = nn.Linear(hidden_size, output_size)
        
        # This is basically our static gain matrix (maybe I should make this a matrix rather than a linear layer...)
        self.k = nn.Linear(input_size, output_size, bias=False) 
        
        # Required for the training
        self.saved_log_probs = []
        self.rewards = []
    
    def forward(self, x):
        # Gating
        g = torch.sigmoid(self.r_linear1(x))
        g = torch.sigmoid(self.r_linear2(g))
        d = softmax(g, dim=-1)

        
        # Swingup
        s = torch.relu(self.s_linear1(x))
        ys = self.s_linear2(s)
        
        # Balance
        yb = self.k(x)
    
        return ys, yb, d
    


#net = Router(2,4,4)
#(ys, yb, d) = net(torch.randn(2))
#print("g1: ", g1)
#print("g2: ", g2)
#print("d: ", d)
#print("x:", x)
#print()
#print("G1", net.r_linear1)
#print("G2", net.r_linear2)


#action = select_action(np.random.randn(2),net)


# In[21]:


# def fixed_step(self,u):
#     th, thdot = self.state # th := theta

#     g = 10.
#     m = 1.
#     l = 1.
#     dt = self.dt

#     u = np.clip(u, -self.max_torque, self.max_torque)
#     self.last_u = u # for rendering
#     costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

#     newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
#     newth = th + newthdot*dt
#     newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

#     self.state = np.array([newth, newthdot])
#     return self._get_obs(), -costs, False, {}



# In[32]:



if __name__ == '__main__':
        
    def select_action(x, policy):
            x = torch.from_numpy(x).float().unsqueeze(0)

            ys, yb, d = policy(x)
            m = torch.distributions.Categorical(d)
            path = m.sample()

            policy.saved_log_probs.append(m.log_prob(path))

            if path.item() == 0:
                return ys.item()
            else:
                return yb.item()

        
        
        
    # Calculates the time weighted rewards, policy losses, and optimizers
    def finish_episode(policy):
        R = 0
        policy_loss = []
        rewards = []

        gamma = .5
        for r in policy.rewards[::-1]:
            R = r + gamma*R
            rewards.append(R)

        rewards = rewards[::-1]
        rewards = torch.tensor(rewards)

        std = rewards.std()
        if torch.isnan(std):
            std = 1

        rewards = (rewards - rewards.mean())/(std + eps)

        for log_prob, reward in zip(policy.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)

        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()

        del policy.rewards[:]
        del policy.saved_log_probs[:]


    
    # Just Naive REINFORCE for pendulum env. 
    env = gym.make('Pendulum-v0')

    #env.step = fixed_step.__get__(env, gym.Env)
    #env.seed(args.seed)
    #torch.manual_seed(args.seed)

    policy = Router(3,8,5)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)



    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):
            action = select_action(state, policy)
            state,reward, done, _ = env.step(np.array([action, 0]))

            policy.rewards.append(reward)
            if done:
                break

            running_reward = running_reward * 0.99 + t * 0.01
            finish_episode(policy) 


            log_interval = 10

            if i_episode % log_interval == 0:
                print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                    i_episode, t, running_reward))
            #if running_reward > env.spec.reward_threshold:
            #    print("Solved! Running reward is now {} and "
            #          "the last episode runs to {} time steps!".format(running_reward, t))
            #    break

