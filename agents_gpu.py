import numpy as np
import torch.nn as nn
import torch as tr
import torch.nn.functional as F

def cross_entropy_with_soft_label(pred, targ):
    return -(targ * pred.log()).sum(dim=-1).mean()

class StepAgent():
    def __init__(self, lr = 0.2, gamma = 0.90, state_shape = (11,11,3)):
        self.gamma = gamma
        self.policy = DeepQNet(state_shape).cuda()
        self.optimizer = tr.optim.SGD(self.policy.parameters(), lr=lr)
        self.states = []
        self.last_action_value = None
        self.rewards = []
        self.dones = []
        self.loss = []
        self.trainingprocess = 0
        
    def reset(self):
        self.last_action_value = None
        self.states = []
        self.rewards = []
        self.dones = []
    
    def act(self, states, definitive = False):
        states = tr.tensor(states).float().cuda()
        prs = self.policy(states)
        if definitive:
            a = prs.max(1)[1].view([-1, 1]).detach()
        else:
            if self.trainingprocess < 1:
                a = tr.multinomial(tr.softmax(5*self.trainingprocess*prs[:,1:], 1), 1).detach()
                a += 1
            else:
                a = prs.max(1)[1].view([-1, 1]).detach()
        self.last_action_value = prs.gather(1, a)
        return a
      
    def reward(self, rewards, states, dones):
        rewards = tr.tensor(rewards).float().cuda()
        states = tr.tensor(states).float().cuda()
        dones = tr.tensor(dones).float().cuda()
        self.rewards.append(rewards)
        self.states.append(states)
        self.dones.append(dones)
        
    def loadnet(self, state_dict):
        self.policy.load_state_dict(state_dict)
    
    def train(self):
        next_prs = self.policy(self.states[-1]).detach()
        next_value = (next_prs[:,1:] * tr.softmax(next_prs[:,1:], 1)).sum(1)*self.gamma*self.trainingprocess + self.rewards[-1]
        
        criterion = nn.MSELoss()
        loss = criterion(self.last_action_value.view(-1),\
                         next_value)
        self.loss.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.trainingprocess = np.minimum(self.trainingprocess + 5e-5, 1)
        
    
class DeepQNet(nn.Module): # if we view it as function Q
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
        self.act = nn.ReLU()
        self.maxpool = nn.MaxPool2d(11)
        self.fc = nn.Linear(32, 5, bias = False)
        nn.init.uniform_(self.fc.weight, a = -0.01, b = 0.01)
        
    def forward(self, state):
        x = self.act(self.bn1(self.conv1(state)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.maxpool(x).reshape([state.shape[0], 32])
        pr_a = self.fc(x)
    
        return pr_a
    
    
#########################################################
######                                              #####
######  I need your help to finish an auto-encoder  #####
######                                              #####
#########################################################
class DeepToMNet(nn.Module):
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
        self.act = nn.ReLU()
        self.maxpool = nn.MaxPool2d(11)
        self.fc = nn.Linear(32, 5, bias = False)
        nn.init.uniform_(self.fc.weight, a = -0.01, b = 0.01)
        
    def forward(self, state):
        x = self.act(self.bn1(self.conv1(state)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        #x = self.bn5(self.maxpool(x).reshape([obs.shape[0], 32]))
        #pr_a = self.bn6(self.fc(x))
        x = self.maxpool(x).reshape([state.shape[0], 32])
        pr_a = self.fc(x)
    
        return pr_a