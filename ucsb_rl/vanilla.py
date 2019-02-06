
# coding: utf-8

# In[1]:


import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import matplotlib.pyplot as plt


# I guess we'll start with a categorical policy
# TODO investigate the cost of action.detach.numpy() and torch.Tensor(state)
def select_action(policy, state):
    m = Categorical(policy(torch.Tensor(state)))
    action = m.sample()
    logprob = m.log_prob(action)

    return action.detach().numpy(), logprob


def main():

    env_name = 'CartPole-v0'
    env = gym.make(env_name)


    # Hard coded policy for the cartpole problem
    # Will eventually want to build up infrastructure to develop a policy depending on:
    # env.action_space
    # env.observation_space

    policy = nn.Sequential(
        nn.Linear(4, 12),
        nn.ReLU(),
        nn.Linear(12,12),
        nn.ReLU(),
        nn.Linear(12,2),
        nn.Softmax(dim=-1)
        )

    optimizer = optim.Adam(policy.parameters(), lr = .1)
    policy(torch.randn(1,4))

    #def vanilla_policy_grad(env, policy, optimizer):

    action_list = []
    state_list = []
    logprob_list = []
    reward_list = []

    avg_reward_hist = []

    num_epochs = 10000
    #batch_size = 20 # how many steps we want to use before we update our gradients
    num_steps = 100 # number of steps in an episode (unless we terminate early)

    loss = torch.zeros(1,requires_grad=True)

    for epoch in range(num_epochs):

        # Probably just want to preallocate these with zeros, as either a tensor or an array
        loss_hist = []
        episode_length_hist = []
        action_list = []
        total_steps = 0

        while True:

            state = env.reset()
            logprob_list = []
            reward_list = []
            action_list = []

            for t in range(num_steps):

                action, logprob = select_action(policy, state)
                state, reward, done, _ = env.step(action.item())

                logprob_list.append(-logprob)
                reward_list.append(reward)
                action_list.append(action)
                total_steps += 1

                if done:
                    break

            # Now Calculate cumulative rewards for each action
            episode_length_hist.append(t)
            reward_ar = np.array(reward_list)
            logprob_ar = np.array(logprob_list)

            episode_loss = [np.sum(reward_ar[i:]*logprob_ar[i:]) for i in range(len(reward_list))]

            avg_reward_hist.append(sum(episode_length_hist) / len(episode_length_hist))
            # loss = torch.sum(torch.stack(loss_hist))
            for action in episode_loss:
                action.backward(retain_graph = True)
                optimizer.step()

    print('hello')
    return policy


def render_loop(env, policy):
    while True:
        state = env.reset()
        cum_rewards = 0

        while True:
            action, _ = select_action(policy,state)
            state, reward, done, _ = env.step(action.item())
            env.render()

            cum_rewards += reward
            if done:
                print('summed reward for espisode: ', cum_rewards)
                break


if __name__ == '__main__':
    main()
