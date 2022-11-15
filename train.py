import env
import numpy as np
import agents_gpu as agents
import copy
import torch as tr
from torch import nn
from matplotlib import pyplot as plt

#e=env.GridWorld(size = 11, n_target = 10)
batch_size = 128

sa = agents.SimpleAgent(lr = 0.2, state_shape = (11, 11, 4))
# sa.loadnet(tr.load('valuenetdict.pt'))

envs = [env.GridWorld(size = 11, n_target = 10) for i in range(batch_size)]

rewardss = []
ws = []
for i in range(200):
    envs = [env.GridWorld(size = 11, n_target = 10) for i in range(batch_size)]
    ws.append(sa.policy.fc.weight[0,0].item())
    obs = np.zeros([batch_size, 2, 11, 11, 4])
    for ie, e in enumerate(envs):
        e.reset()
        obs[ie] = e.observe()
    for istep in range(30):
        states = tr.tensor(obs[:, 0, :, :, :]).permute([0, 3, 1, 2]).float().cuda()
        act = sa.act(states)
        # acts.append(act)
        
        obs = np.zeros([batch_size, 2, 11, 11, 4])
        rewards = np.zeros([batch_size])
        dones = np.zeros([batch_size])
        for ie, e in enumerate(envs):
            _obs, reward, done = e.step([act[ie, 0].item(), 0])
            obs[ie,:,:,:,:] = _obs
            rewards[ie] = reward
            dones[ie] = done
        re = np.sum(rewards)
        rewards = tr.tensor(rewards).float().cuda()
        dones = tr.tensor(dones).float().cuda()
        #if done:
        #    break
        sa.train(rewards)
    sa.reset()
    rewardss.append(re)
    print("Epoch {}: reward {}".format(i, re) )
# print(np.array([i for i in zip(rewards.cpu().detach().numpy(), act.cpu().detach().numpy()[:,0])]).T)

tr.save(sa.value.state_dict(), 'valuenetdict2.pt')