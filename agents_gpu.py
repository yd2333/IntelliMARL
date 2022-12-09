import numpy as np
import torch as tr
import torch.nn as nn
import torch.nn.functional as F

def cross_entropy_with_soft_label(pred, targ):
    return -(targ * pred.log()).sum(dim=-1).mean()

class ExToMAgent():
    def __init__(self, lr = 0.2, gamma = 0.90, obs_shape = (11,11,4)):
        self.gamma = gamma
        self.policy = DeepQNet(obs_shape).cuda()
        # self.policy = None
        # self.ToMNet = DeepToMNet((11,11,3)).cuda()
        # self.ToMNet = ConvToMNet((11,11,3)).cuda()
        self.ToMNet = ConvToMNet((11,11,3)).cuda()
        self.policy_optimizer = tr.optim.SGD(self.policy.parameters(), lr=lr)
        self.ToMNet_optimizer = tr.optim.SGD(self.ToMNet.parameters(), lr=lr)
        self.obs = []
        self.last_action_value = None
        self.rewards = []
        self.dones = []
        self.loss = []
        self.ToMloss = []
        self.trainingprocess = 0
        
    def reset(self):
        self.last_action_value = None
        self.obs = []
        self.rewards = []
        self.dones = []
    
    def act(self, obs, definitive = False):
        obs = tr.tensor(obs).float().cuda()
        with tr.no_grad():
            obs[:,3,:,:] = self.ToMNet(obs[:,[0, 1, 3],:,:])
        prs = self.policy(obs)
        if definitive:
            a = prs.max(1)[1].view([-1, 1]).detach()
        else:
            if self.trainingprocess < 1: # exploration
                a = tr.multinomial(tr.softmax(5*self.trainingprocess*prs[:,1:], 1), 1).detach()
                a += 1
            else: # expoitation 
                a = prs.max(1)[1].view([-1, 1]).detach() 
        self.last_action_value = prs.gather(1, a)
        return a
      
    def record(self, rewards, obs, dones):
        rewards = tr.tensor(rewards).float().cuda()
        obs = tr.tensor(obs).float().cuda()
        dones = tr.tensor(dones).float().cuda()
        self.rewards.append(rewards)
        self.obs.append(obs)
        self.dones.append(dones)
        
    def load_policy(self, policy_name="policy"):
        path = f"./model/{policy_name}.pt"
        self.policy.load_state_dict(tr.load(path))
    
    def save_policy(self, name="policy"):
        policy_path = f"./model/{name}.pt"
        tr.save(self.policy.state_dict(), policy_path)

    def train_policy(self):
        obs = self.obs[-1]
        with tr.no_grad():
            obs[:,3,:,:] = self.ToMNet(obs[:,[0, 1, 3],:,:])
        prs = self.policy(obs)
        next_prs = self.policy(obs).detach()
        # print(f"policy_return: {next_prs[0]}")
        next_value = (next_prs[:,1:] * tr.softmax(next_prs[:,1:], 1)).sum(1)*self.gamma*self.trainingprocess + self.rewards[-1]
        # print("next value", next_value.shape)
        criterion = nn.MSELoss()
        loss = criterion(self.last_action_value.view(-1),\
                         next_value)
        self.loss.append(loss.item())
        self.policy_optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy_optimizer.step()
        self.trainingprocess = np.minimum(self.trainingprocess + 1.01e-3, 1) ## prune
        self.rewards = []
        self.obs = []
        self.dones = []

        
    def train_ToMNet(self, obs, trajectories):
        obs = tr.tensor(obs).float().cuda()
        trajectories = tr.tensor(trajectories).float().cuda()
        pred_trajs = self.ToMNet(obs).clamp(1e-5, 1-1e-5)
        
        #criterion = nn.L1Loss()
        #loss = criterion(trajectories, pred_trajs)
        loss = -tr.mean(trajectories*tr.log(pred_trajs) + (1-trajectories)*tr.log(1-pred_trajs))
        self.ToMloss.append(loss.item())
        loss.backward()
        for param in self.ToMNet.parameters():
            param.grad.data.clamp_(-1, 1)
        self.ToMNet_optimizer.step()

    def save_models(self, policy_name = "policy", tom_name = "tom"):
        policy_path = f"./model/{policy_name}.pt"
        tom_path = f"./model/{tom_name}.pt"
        tr.save(self.policy.state_dict(), policy_path)
        tr.save(self.ToMNet.state_dict(), tom_path)

    def load_models(self, policy_name = "policy", tom_name = "tom"):
        policy_path = f"./model/{policy_name}.pt"
        tom_path = f"./model/{tom_name}.pt"
        self.policy.load_state_dict(tr.load(policy_path))
        self.ToMNet.load_state_dict(tr.load(tom_path))
    
    
class ConvToMNet(nn.Module):
    def __init__(self, state_shape = (16,16,3)):
        super().__init__()

        self.conv1 = nn.Conv2d(state_shape[-1], 16, stride = 1, kernel_size = 3, padding = 1, bias = False)
        nn.init.uniform_(self.conv1.weight, a = -0.01, b = 0.01)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, stride = 1, kernel_size = 5, padding = 2, bias = False)
        nn.init.uniform_(self.conv2.weight, a = -0.01, b = 0.01)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, stride = 1, kernel_size = 7, padding = 3, bias = False)
        nn.init.uniform_(self.conv3.weight, a = -0.01, b = 0.01)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, stride = 1, kernel_size = 9, padding = 4, bias = False)
        nn.init.uniform_(self.conv4.weight, a = -0.01, b = 0.01)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 1, stride = 1, kernel_size = 9, padding = 4, bias = False)
        nn.init.uniform_(self.conv5.weight, a = -0.01, b = 0.01)
        self.bn5 = nn.BatchNorm2d(1)
        
        self.act = nn.ReLU()
        
    def forward(self, obs):
        x = self.act(self.bn1(self.conv1(obs)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = tr.sigmoid(self.bn5(self.conv5(x))).reshape([-1, 11, 11])
    
        return x

class DeepQNet(nn.Module): # if we view it as function Q
    def __init__(self, obs_shape = (11,11,4)): 
        super().__init__()
        self.conv1 = nn.Conv2d(obs_shape[-1], 16, stride = 1, kernel_size = 3, padding = 1, bias = False)
        nn.init.uniform_(self.conv1.weight, a = -0.01, b = 0.01)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, stride = 1, kernel_size = 5, padding = 2, bias = False)
        nn.init.uniform_(self.conv2.weight, a = -0.01, b = 0.01)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, stride = 1, kernel_size = 7, padding = 3, bias = False)
        nn.init.uniform_(self.conv3.weight, a = -0.01, b = 0.01)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, stride = 1, kernel_size = 9, padding = 4, bias = False)
        nn.init.uniform_(self.conv4.weight, a = -0.01, b = 0.01)
        self.bn4 = nn.BatchNorm2d(32)
        self.act = nn.ReLU()
        self.maxpool = nn.MaxPool2d(11)
        self.fc = nn.Linear(32, 5, bias = False)
        nn.init.uniform_(self.fc.weight, a = -0.01, b = 0.01)
        
    def forward(self, obs):
        x = self.act(self.bn1(self.conv1(obs)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.maxpool(x).reshape([obs.shape[0], 32])
        pr_a = self.fc(x)
        return pr_a

    