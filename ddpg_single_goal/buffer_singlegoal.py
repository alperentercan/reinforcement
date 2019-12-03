import numpy as np
import torch

class Replay_buffer():
    def __init__(self,size,obs_size,action_size):
        self.size = size
        self.obs_size = obs_size
        self.action_size = action_size
        self.width = obs_size + action_size + 1 + obs_size + 1
        self.replay_buffer = torch.zeros([self.size,self.width],dtype=torch.float32)
        self.buff_ind = 0
        self.buff_curr_size = 0
        
    
    def add_entry(self,obs,a,reward,obs_new,done):
        done = 0 if done else 1
        transition_entry = torch.tensor(np.hstack((obs, a, reward, obs_new,done)))
        self.replay_buffer[self.buff_ind,:] = transition_entry
        self.buff_ind = (self.buff_ind + 1)%self.size;
        self.buff_curr_size = min(self.size,self.buff_curr_size + 1);
    
      
    def sample_batch(self,batch_size):
        minibatch_size = min(batch_size,self.buff_curr_size)
        batch = self.replay_buffer[np.random.choice(self.buff_curr_size,minibatch_size),:]
        return batch
    
    def sample_split_batch(self,batch_size):
        minibatch_size = min(batch_size,self.buff_curr_size)
        batch_ind = np.random.choice(self.buff_curr_size,minibatch_size)
        obs_batch = self.replay_buffer[batch_ind,:self.obs_size]#.reshape(-1,1)
        act_batch = self.replay_buffer[batch_ind,self.obs_size:self.obs_size + self.action_size]#.reshape(-1,1)
        reward_batch = self.replay_buffer[batch_ind,self.obs_size + self.action_size:self.obs_size + self.action_size + 1]#.reshape(-1,1)
        obs_new_batch = self.replay_buffer[batch_ind, -self.obs_size-1:-1]#.reshape(-1,1)
        done_batch = self.replay_buffer[batch_ind,-1].reshape(-1,1)
        
        return obs_batch,act_batch,reward_batch,obs_new_batch,done_batch