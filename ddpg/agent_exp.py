
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from buffer_singlegoal import Replay_buffer
from util import *

# from util import to_tensor
# from util import hard_update
from random_process import OrnsteinUhlenbeckProcess
# from memory import SequentialMemory
from model import (Actor, Critic)
criterion = nn.MSELoss()

class Agent():
    def __init__(self,discount):
        self.discount = discount
        self.obs = None
        self.reward = None
        self.a = None
        self.done = None
        self.new_obs = None
        
    def observe(self,obs,reward,done):
        self.new_obs = obs
        self.reward = reward
        self.done = done
        
        
    def select_action(self):
        return None

class DDPG(object):
    def __init__(self,nb_states,nb_actions,args):
        
# #         super().__init__(args.discount)
#         self.discount = args.discount
#         #Functions
#         self.loss_func = torch.nn.MSELoss()
        
#         #Hyper Parameters
# #         self.epsilon = 0.3
#         self.tau = args.tau # 1-stability
#         self.replay_buffer_size = args.rbsize
#         self.minibatch_size_for_replay = args.bsize
#         self.lr_critic = args.lr_cr
#         self.lr_actor = args.lr_ac
        
#         #Network_parameters
#         ah1 = 400
#         ah2 = 300
#         ch1 = 400
#         ch2 = 300
        
#         #Env parameters lookup
#         self.obs_size = obs_size
#         self.action_size =  action_size
#         self.goal_size = 0
#         self.action_space_range = action_range
        
#         # Other parameters
#         n_inputs_critic = obs_size + action_size #+ goal_size
#         n_inputs_actor = obs_size # + goal_size
#         n_outputs_actor = action_size
# #         self.buffer = Replay_buffer(self.replay_buffer_size,self.obs_size,self.action_size)
#         self.memory = SequentialMemory(limit=args.rbsize, window_length=args.window_length)
 
#         self.critic_optim  = torch.optim.Adam(self.critic.parameters(), lr=args.rate)

#         hard_update(self.target_actor, self.actor) # Make sure target is with the same weight
#         hard_update(self.target_critic, self.critic)
          
# #         self.optim_critic= torch.optim.Adam(self.critic.parameters(), lr = self.lr_critic)
# #         self.optim_actor= torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
#         self.random_process = OrnsteinUhlenbeckProcess(size=self.action_size, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)
#         self.depsilon = 1/50000
#         self.epsilon = 1.0
#         self.is_training=True


        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions= nb_actions
        
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2, 
            'init_w':args.init_w
        }
        
        n_inputs_critic = nb_states + nb_actions #+ goal_size
        n_inputs_actor = nb_states # + goal_size
        n_outputs_actor = nb_actions
        
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)

        
#         self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
#         self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)

#         self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
#         self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)

    
    # Initialize target network weights to behavior   
#         for i in [0,2,4]:
#             self.critic_target[i].weight.data = self.critic[i].weight.data.clone()
#         for i in [0,2,4]:
#             self.actor_target[i].weight.data = self.actor[i].weight.data.clone()
#         critic_target.zero_grad()
#         actor_target.zero_grad()
         
        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        self.buffer = Replay_buffer(args.rmsize,self.nb_states,self.nb_actions)
        #Create replay buffer
#         self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        # 
        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.new_episode = False
        self.is_training = True

        # 
#         if USE_CUDA: self.cuda()

  
    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(
            self.actor(to_tensor(np.array([s_t])))
        ).squeeze(0)
        action += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon
        self.a_t = action
        return action
    
#     def observe(self,obs,reward,done):
#         super().observe(obs,reward,done)
#         self.buffer.add_entry(self.obs,self.a,self.reward,self.new_obs,int(done))

    def observe(self,reward,obs,done):
#         super().observe(obs,reward,done)
#         self.buffer.add_entry(self.obs,self.a,self.reward,self.new_obs,int(done)) 

        if self.is_training:    
            if self.new_episode:
                self.new_episode = False
                self.s_t = obs
            else:
                if done:
                    self.new_episode = True
                self.buffer.add_entry(self.s_t, self.a_t, reward,obs, done)
#                 self.memory.append(self.s_t, self.a_t, reward, done)

                self.s_t = obs
#         if self.is_training:    
#             self.memory.append(self.s_t, self.a_t, reward, done)
#             self.s_t = obs
            

    def random_action(self):
        action = np.random.uniform(-1.,1.,self.nb_actions)#*self.action_space_range
        self.a_t = action
        return action
    
#     def select_action(self, obs, decay_epsilon=True):
#         if np.random.uniform() > self.epsilon:
#             with torch.no_grad():
#                 a = self.actor(torch.tensor(obs,dtype=torch.float32)).detach().numpy()
#                 a = a*(self.action_space_range.numpy())

#                 a = a + np.random.normal(0,self.action_space_range/4,a.shape)
#         else:
#             a = np.random.uniform(-self.action_space_range,self.action_space_range,self.action_size)
        
#         self.a = a
#         return a
    
#     def update_policy(self):

#         bobs,bact,br,bobs2,bd = self.buffer.sample_split_batch(self.minibatch_size_for_replay)
#         bdc = np.abs(bd-1)
#         target_critic_input = torch.cat([bobs2, self.target_actor(bobs2)],dim=1)#*self.action_space_range
#         Z = self.target_critic(target_critic_input)
#         Y = br + self.discount*bdc*Z

#         # Update Critic
#         self.critic.zero_grad()
#         pred = self.critic(torch.cat([bobs,bact],dim=1))
#         loss_critic = self.loss_func(pred,Y)
#         loss_critic.backward()
#         self.optim_critic.step()

#          # Update Actor
#         self.actor.zero_grad()
#         actor_loss = -self.critic(torch.cat([bobs,self.actor(bobs)],dim=1)).mean()#*self.action_space_range
#         actor_loss.backward()             
#         self.optim_actor.step()  
#         self.soft_update()
        
    ### From DDPG IMPLEMENTATION
    def update_policy(self):
#         # Sample batch
#         state_batch, action_batch, reward_batch, \
#         next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.buffer.sample_split_batch(self.batch_size)
        
#         state_batch = np.array(state_batch)
#         action_batch = np.array(action_batch)
#         reward_batch = np.array(reward_batch)
#         next_state_batch = np.array(next_state_batch)
#         terminal_batch = np.array(terminal_batch)


#         state_batch = torch.tensor(state_batch,dtype=torch.float32)
#         action_batch = torch.tensor(action_batch,dtype=torch.float32)
#         reward_batch = torch.tensor(reward_batch,dtype=torch.float32)
#         next_state_batch = torch.tensor(next_state_batch,dtype=torch.float32)
#         terminal_batch = torch.tensor(terminal_batch,dtype=torch.float32)

        state_batch = state_batch.clone().detach()#torch.tensor(state_batch,dtype=torch.float32)
        action_batch = action_batch.clone().detach()#torch.tensor(action_batch,dtype=torch.float32)
        reward_batch = reward_batch.clone().detach()#torch.tensor(reward_batch,dtype=torch.float32)
        next_state_batch = next_state_batch.clone().detach()#torch.tensor(next_state_batch,dtype=torch.float32)
        terminal_batch = terminal_batch.clone().detach()#torch.tensor(terminal_batch,dtype=torch.float32)

#         bdc = np.abs(bd-1)
        with torch.no_grad():
#             target_critic_input = torch.cat([next_state_batch, self.actor_target(next_state_batch)],dim=1)#*self.action_space_range
            Z = self.critic_target([next_state_batch,self.actor_target(next_state_batch)])

            intermediary = self.discount*terminal_batch*Z

            Y = reward_batch + intermediary
        # Update Critic
        self.critic.zero_grad()
        pred = self.critic([state_batch,action_batch])
        loss_critic = criterion(pred,Y)
        loss_critic.backward()
        self.critic_optim.step()
#         ## Important
        self.critic.zero_grad()
         # Update Actor
        self.actor.zero_grad()
        actor_loss = -self.critic([state_batch,self.actor(state_batch)]).mean()#*self.action_space_range
        actor_loss.backward()             
        self.actor_optim.step()  
        
#         # Prepare for the target q batch
#         next_q_values = self.critic_target(torch.cat([
#             to_tensor(next_state_batch, volatile=True),
#             self.actor_target(to_tensor(next_state_batch, volatile=True)),
#         ],dim=1)) ## Check
#         next_q_values.volatile=False

#         target_q_batch = to_tensor(reward_batch) + \
#             self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values

#         # Critic update
#         self.critic.zero_grad()

#         q_batch = self.critic(torch.cat([ to_tensor(state_batch), to_tensor(action_batch) ],dim=1))
#         value_loss = criterion(q_batch, target_q_batch)
        
# #         value_loss = self.loss_func(target_q_batch,q_batch) ### Attention
#         value_loss.backward()
#         self.critic_optim.step()

#         # Actor update
#         self.actor.zero_grad()

#         policy_loss = -self.critic(torch.cat([
#             to_tensor(state_batch),
#             self.actor(to_tensor(state_batch))
#         ],dim=1))

#         policy_loss = policy_loss.mean()
#         policy_loss.backward()
#         self.actor_optim.step()

#         # Target update
#         soft_update(self.actor_target, self.actor, self.tau)
#         soft_update(self.critic_target, self.critic, self.tau)
        self.soft_update()
        

    def soft_update(self):
        self.critic_target.soft_update(self.critic,0.001)
        self.actor_target.soft_update(self.actor,0.001)

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()


#     def soft_update(self):      
#         #Update target networks          
#         with torch.no_grad():
#             for i in [0,2,4]:
#                 self.critic_target[i].weight.data = (self.tau*self.critic[i].weight.data.clone() + 
#                                                      (1-self.tau) *self.critic_target[i].weight.data.clone())
#             for i in [0,2,4]:
#                 self.actor_target[i].weight.data = (self.tau*self.actor[i].weight.data.clone() + 
#                                                     (1-self.tau)*self.actor_target[i].weight.data.clone())
                
    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )


    def save_model(self,output):
        return 0
#         torch.save(
#             self.actor.state_dict(),
#             '{}/actor.pkl'.format(output)
#         )
#         torch.save(
#             self.critic.state_dict(),
#             '{}/critic.pkl'.format(output)
#         )

#     def seed(self,s):
#         torch.manual_seed(s)
#         if USE_CUDA:
#             torch.cuda.manual_seed(s)
