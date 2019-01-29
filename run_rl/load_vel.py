
# coding: utf-8


import torch
import pprint
import vel
from vel.rl.models.policy_gradient_model_separate import PolicyGradientModelSeparateFactory
from vel.rl.models.backbone.mlp import MLPFactory
from vel.util.random import set_seed
from vel.rl.env.mujoco import MujocoEnv



state_dict = torch.load('/Users/sgillen/work_dir/output/checkpoints/walker_ppo/0/checkpoint_00000489.data', map_location = 'cpu')
hidden_dict =  torch.load('/Users/sgillen/work_dir/output/checkpoints/walker_ppo/0/checkpoint_hidden_00000489.data', map_location = 'cpu')


seed = 1002
set_seed(seed) # Set random seed in python std lib, numpy and pytorch
env = MujocoEnv('Walker2d-v2').instantiate(seed=seed)


policy_in_size = state_dict['policy_backbone.model.0.weight'].shape[1]
value_in_size = state_dict['policy_backbone.model.0.weight'].shape[1]


model_factory = PolicyGradientModelSeparateFactory(
    policy_backbone=MLPFactory(input_length=17, hidden_layers=[64, 64], activation='tanh'),
    value_backbone=MLPFactory(input_length=17, hidden_layers=[64, 64], activation='tanh'),
)

#sgillen - pretty sure this takes care of the output sizes for us automatically
model = model_factory.instantiate(action_space=env.action_space)
model.load_state_dict(state_dict)

env.allow_early_resets = True

ob = env.reset()

while True:
    #action = model.step(torch.Tensor(ob)).detach().numpy()
    action = model.step(torch.Tensor(ob).view(1,-1))['actions'].detach().numpy()
    ob, _, done, _ =  env.step(action)
    #if reward == 1:
    #    print("balanced")
    env.render()
    if done:
        ob = env.reset()

