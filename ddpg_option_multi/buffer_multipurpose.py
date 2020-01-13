import numpy as np
import torch

class Replay_buffer():
    def __init__(self,size,partition=[]):
        self.size = size
#         self.obs_size = obs_size
#         self.action_size = action_size
        self.partition = partition
        self.width = sum(partition)
        self.replay_buffer = torch.zeros([self.size,self.width],dtype=torch.float32)
        self.buff_ind = 0
        self.buff_curr_size = 0
        
    
    def add_entry(self,obs):
        if len(obs) != len(self.partition):
            print(f'Obs expected to be of length {len(self.partition)} instead found {len(obs)}')
        else:
#             done = 0 if done else 1
#             transition_entry = torch.tensor(np.concatenate([obs, a,w, reward, obs_new,done],axis=None))
            transition_entry = torch.tensor([i for j in obs for i in j])
#             print(f'Transition_entry is of size {transiton_entry.shape}')
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
        
        output = []
        for i in range(len(self.partition)):
            output.append(self.replay_buffer[batch_ind,sum(self.partition[:i]):sum(self.partition[:i+1])])
            

#         obs_batch = self.replay_buffer[batch_ind,:self.obs_size]#.reshape(-1,1)
#         act_batch = self.replay_buffer[batch_ind,self.obs_size:self.obs_size + self.action_size]#.reshape(-1,1)
#         w_batch = self.replay_buffer[batch_ind,self.obs_size + self.action_size:self.obs_size + self.action_size + 1]
#         reward_batch = self.replay_buffer[batch_ind,self.obs_size + self.action_size +1:self.obs_size + self.action_size + 2]
#         obs_new_batch = self.replay_buffer[batch_ind, -self.obs_size-1:-1]#.reshape(-1,1)
#         done_batch = self.replay_buffer[batch_ind,-1].reshape(-1,1)
        
        return output
