import numpy  as np
import copy

def accessible(agentloc, wall, targetloc): ## not useful anymore
    x, y = agentloc
    xt, yt = targetloc
    if wall[x, y] or wall[xt, yt]:
        return False
    if agentloc == targetloc:
        return True
    wall[x, y] = 1
    return accessible([x-1, y], copy.deepcopy(wall), targetloc) or\
           accessible([x+1, y], copy.deepcopy(wall), targetloc) or\
           accessible([x, y-1], copy.deepcopy(wall), targetloc) or\
           accessible([x, y+1], copy.deepcopy(wall), targetloc)

def connectivity(walls): ## walls not separating the world
    labelsets = []
    labelmat = np.ones_like(walls)*np.inf
    label = 1
    for i in range(walls.shape[0]):
        for j in range(walls.shape[1]):
            if walls[i,j]:
                continue
            elif (walls[i-1, j]) and (walls[i, j-1]):
                labelmat[i, j] = label
                labelsets.append({label})
                label = label + 1
            else:
                nblabels = [labelmat[i-1,j], labelmat[i,j-1]]
                labelmat[i,j] = np.min(nblabels)
                for lset in labelsets:
                    if nblabels[0] in lset or nblabels[1] in lset:
                        for lb in nblabels:
                            if lb != np.inf:
                                lset.add(lb)
                        break
    for i in range(walls.shape[0]):
        for j in range(walls.shape[1]):
            if walls[i,j]:
                continue
            else:
                for lset in labelsets:
                    if labelmat[i,j] in lset:
                        labelmat[i,j] = min(lset)
                        break
                        
    return np.unique(labelmat).shape[0] == 2
    
                        
                
                
class GridWorld:
    def __init__(self, size = 11, n_agent = 2, n_wall = 2, n_target = 1, remember_states = True):
        '''
        World:
        0 : wall
        1 : target
        
        Agent:
        {
        0 : Location
        }
        
        Action:
        0 : none
        1 : left
        2 : right
        3 : up
        4 : down
        '''
        self.world_shape = (size, size)
        self.size = size # the length of the 2d grid 
        self.n_wall = n_wall
        self.n_agent = n_agent
        self.n_target = n_target
        self.remember_states = remember_states
        self.init_state = {'world' : np.empty([size, size, 2]),\
                      'agents' : np.empty([size, size, n_agent]),\
                      'done' : False}
        self.state = {'world' : np.empty([size, size, 2]),\
                      'agents' : np.empty([size, size, n_agent]),\
                      'done' : False}
        
        wall = self.gen_wall() # (11,11) grid world
        targets = self.gen_targets(wall)  # (11,11) grid world
        agents = self.gen_agents(wall, targets)
            
        # target = np.any(targets, 2)
        self.init_state['world'] = np.stack([wall, targets], 2)
        self.init_state['agents'] = agents
        self.reset()

    def gen_wall(self):
        wall = np.zeros(self.world_shape)
        wall[0, :] = 1 # 1st row
        wall[self.size-1, :] = 1 # last row
        wall[:, 0] = 1 # 1st column 
        wall[:, self.size-1] = 1 # last column 
        for _ in range(self.n_wall):
            direction = np.random.choice([0, 1] , 1)[0] # generate 1 or 0 randomly 
            st = np.random.randint(self.size, size=2) # output shape is 2
            if st[direction] == 10:
                length = 1
            else:
                length = np.random.randint(low=1, high=self.size - st[direction] + 1)
            if direction == 0:
                wall[st[0]:st[0]+length, st[1]] = 1
            else:
                wall[st[0], st[1]:st[1]+length] = 1
        if connectivity(wall):
            return wall
        else:
            return self.gen_wall()
            
    # generate reward where there is no wall existed
    def gen_targets(self, wall):
        slots = np.where(np.logical_not(wall.flatten()))[0].tolist()
        targets = np.zeros(self.world_shape).flatten()
        for i_target in range(self.n_target):
            il = slots[np.random.randint(len(slots))]
            targets[il] = 1
            slots.remove(il)
        targets = targets.reshape(self.world_shape)
        return targets
    
    def gen_agents(self, wall, targets):
        slots = np.logical_not(wall + targets).flatten()
        slots = np.where(slots)[0].tolist()
        agents = np.zeros(self.world_shape + (self.n_agent,))
        for i_agent in range(self.n_agent):
            agent = np.zeros(self.world_shape).flatten()
            il = slots[np.random.randint(len(slots))]
            agent[il] = 1
            slots.remove(il)
            agents[:,:,i_agent] = agent.reshape(self.world_shape)
        return agents
        
    def reset(self):
        self.state = copy.deepcopy(self.init_state)
        if self.remember_states:
            self.history = []
            state = self.state.copy()
            state['obs'] = self.observe()
            self.history.append(state)
            
        return self.state
        
    def observe(self):
        obs = np.zeros((self.n_agent,)+self.world_shape+(4,)) # 2, 16, 16, 4
        for i_agent in range(self.n_agent):
            world = self.state['world']
            agents = self.state['agents'][:,:,[i for i in range(self.n_agent) if i != i_agent]] #obervasion of other agent
            obs[i_agent, :, :, :2] = world # for agent:i_agent, the first 2/4 tensor are world observasion
            obs[i_agent, :, :, 2]= self.state['agents'][:,:,i_agent] # for the 3rd tensor of agent:i_agent is the one-hot of itself
            obs[i_agent, :, :, 3] = np.any(agents, 2) # the forth tensor is the obsercasion of other agent
        return obs
        
    def step(self, action):
        # action.shape: agent
        for i_agent in range(self.n_agent):
            if np.sum(self.state['agents'][:, :, i_agent]) == 0:
                assert False
                x, y = np.random.randint(0, 11, [2])
            
            [x], [y] = np.where(self.state['agents'][:, :, i_agent])
            attx, atty = x, y
            if action[i_agent] == 0:
                pass
            elif action[i_agent] == 1:
                attx = np.maximum(0, x - 1)
            elif action[i_agent] == 2:
                attx = np.minimum(self.world_shape[0]-1, x + 1)
            elif action[i_agent] == 3:
                atty = np.maximum(0, y - 1)
            elif action[i_agent] == 4:
                atty = np.minimum(self.world_shape[1]-1, y + 1)
            else:
                if i_agent == 0:
                    print(action[i_agent])
                    assert False
            
            wall = self.state['world'][:,:,0]
            if wall[attx, atty]:
                attx, atty = x, y
            
            self.state['agents'][:, :, i_agent] = np.zeros(self.world_shape)
            self.state['agents'][attx, atty, i_agent] = 1
                
        agents = np.any(self.state['agents'], 2)
        target = self.state['world'][:,:,1]
        reward = np.sum(agents * target) - 0 * np.sum(agents*wall)
        
        done = np.logical_not(np.any(self.state['world'][:,:,1]))
        self.state['done'] = done
        self.state['world'][:,:,1] = np.logical_and(target, np.logical_not(agents))
        
        
        obs = self.observe()
        if self.remember_states:
            state = copy.deepcopy(self.state)
            state['obs'] = obs
            self.history.append(state)
        
        return obs, reward, done
        
    def recall(self):
        obs = []
        trajs = []
        traj = np.zeros(self.world_shape[:2]+(self.n_agent,))
        for i, state in enumerate(self.history[::-1]):
            #if state['done']:
            #    continue
            traj += state['agents'][:,:,:]
        obs.append(copy.deepcopy(state['obs']))
        trajs.append(np.float32(copy.deepcopy(traj) > 0))
        _trajs = np.array(trajs)
        trajs = np.zeros((_trajs.shape[0],) + self.world_shape[:2]+(self.n_agent,))
        for ia in range(self.n_agent):
            _ind = np.ones(self.n_agent)
            _ind[ia] = False
            trajs[:,:,:,ia] = np.any(_trajs[:,:,:,np.bool8(_ind)], 3)
        
        return (np.array(obs), trajs)
        
    def __str__(self):
        labelmat = 7*self.state['world'][:,:,0] + 8*self.state['world'][:,:,1]
        for i_agent in range(self.n_agent):
            labelmat = labelmat + (i_agent + 1)*self.state['agents'][:,:,i_agent]
            labelmat[np.logical_and(self.state['world'][:,:,0], self.state['agents'][:,:,i_agent])] = 9
        
        labelmat[labelmat > 9] = 9
        # labelmat[self.state['world'][:,:,0] & self.state['agent'][:,:,i_agent]] = 9
        
        
        statestr = str(labelmat).replace('0', ' ').replace('.', '')
        statestr = statestr.replace('7', '#').replace('8', 'r').replace('9', 'X')
        statestr = statestr.replace('[', ' ').replace(']', ' ')
        
        return statestr
        