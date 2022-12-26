import env
import numpy as np
import agents_gpu as agent
import visualization
import copy
import torch as tr
from torch import nn
from matplotlib import pyplot as plt

n_agents = 2
agents = [agent.ExToMAgent(lr = 0.01, obs_shape = (11, 11, 4)) for i in range(n_agents)]
batch_size = 128
n_runs = 500

rewardss = []
ws = []
for i in range(n_runs):
    envs = [env.GridWorld(size = 11, n_target = 4) for i in range(batch_size)]
    ws.append(copy.deepcopy(agents[0].policy.fc.weight.cpu().detach())) 
    state = np.zeros([batch_size, 2, 11, 11, 4])
    for ie, e in enumerate(envs): # initial env
        e.reset()
        state[ie] = e.observe()
    observes = [state[:, ia, :, :, :].transpose([0, 3, 1, 2]) for ia in range(n_agents)]
    re = []
    
    for istep in range(10):
        act = [agents[ia].act(observes[ia]) for ia in range(n_agents)]
        
        state = np.zeros([batch_size, n_agents, 11, 11, 4])
        rewards = np.zeros([batch_size])
        dones = np.zeros([batch_size])
        for ie, e in enumerate(envs):
            _state, reward, done = e.step([act[ia][ie, 0].item() for ia in range(n_agents)])
            state[ie,:,:,:,:] = _state
            rewards[ie] = reward
            dones[ie] = done
        observes = [state[:, ia, :, :, :4].transpose([0, 3, 1, 2]).copy() for ia in range(n_agents)]
        re.append(np.mean(rewards))

        for ia in range(n_agents):
            agents[ia].record(rewards, observes[ia], dones)
            agents[ia].train_policy()
        if agents[0].trainingprocess==1:
            history_states, trajectories = [], []
            for ie in range(batch_size):
                _history_states, _trajectories = envs[ie].recall()
                history_states.append(_history_states)
                trajectories.append(_trajectories)
            history_obs = np.concatenate(history_states)
            trajectories = np.concatenate(trajectories)
            for ia in range(n_agents):
                obs = history_obs[:, ia, :, :, :][:, :, :, [0, 1, 3]].transpose([0, 3, 1, 2]).copy()
                traj = trajectories[:,:,:,ia]
                for j in range(round(10)):
                    agents[ia].train_ToMNet(obs, traj)
    for ia in range(n_agents):
        agents[ia].reset()
    rewardss.append(re)
    if i%10 ==0:
        if agents[0].trainingprocess==1:
            print("Epoch \t%d \t Reward %.2f \t ToM Loss %.5f"%(i, np.sum(rewardss[-1]), agents[0].ToMloss[-1]))
        else:
            print("Epoch \t%d \t Reward %.2f \t ToM Loss untrained"%(i, np.sum(rewardss[-1])))
    tr.cuda.empty_cache()
tr.save(agents[0].policy.state_dict(), 'PolicyNet_dict_ExToM.pt')
tr.save(agents[0].ToMNet.state_dict(), 'ToMNet_dict.pt')
np.save("ExToM.npy", ((agents[0].loss, agents[1].loss, agents[0].ToMloss, agents[1].ToMloss), rewardss), dtype=object)

visualization.save_visualization_to(agents[0], "ToM_results.png")