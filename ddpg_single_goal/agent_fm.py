
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from buffer_singlegoal import Replay_buffer
from util import *
# from util import to_tensor
# from util import hard_update
from random_process import OrnsteinUhlenbeckProcess
from agent import DDPG
from new_models import Critic_Fm
# from memory import SequentialMemory
from model import Actor
criterion = nn.MSELoss()


class DDPG_FM(DDPG):
    
    def __init__(self,nb_states,nb_actions,args):
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
        
#         self.critic = Critic_Fm(nb_states,nb_actions)
#         self.critic = torch.nn.Sequential(
#         torch.nn.Linear(n_inputs_critic, args.hidden1),
#         torch.nn.ReLU(),
#         torch.nn.Linear(args.hidden1, args.hidden2),
#         torch.nn.ReLU(),       
#         torch.nn.Linear(args.hidden2, 1))


#         self.actor = torch.nn.Sequential(
#         torch.nn.Linear(n_inputs_actor, args.hidden1),
#         torch.nn.ReLU(),
#         torch.nn.Linear(args.hidden1,args.hidden2),
#         torch.nn.ReLU(),       
#         torch.nn.Linear(args.hidden2, n_outputs_actor),
#         torch.nn.Tanh())

        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic_Fm(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic_Fm(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)

#         ### Target Networks
#         self.critic_target = Critic_Fm(nb_states,nb_actions)
#         self.critic_target = torch.nn.Sequential(
#         torch.nn.Linear(n_inputs_critic, args.hidden1),
#         torch.nn.ReLU(),
#         torch.nn.Linear(args.hidden1, args.hidden2),
#         torch.nn.ReLU(),       
#         torch.nn.Linear(args.hidden2, 1))
#           self.actor_target = Actor()
#         self.actor_target = torch.nn.Sequential(
#         torch.nn.Linear(n_inputs_actor, args.hidden1),
#         torch.nn.ReLU(),
#         torch.nn.Linear(args.hidden1, args.hidden2),
#         torch.nn.ReLU(),       
#         torch.nn.Linear(args.hidden2, n_outputs_actor),
#         torch.nn.Tanh())

#         self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)
#         self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)

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
        self.mean_td_error = 1
        self.mean_model_error = 1
        self.update_count = 0

    ### From DDPG IMPLEMENTATION
    def update_policy(self):
#         # Sample batch
#         state_batch, action_batch, reward_batch, \
#         next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.buffer.sample_split_batch(self.batch_size)
        

        state_batch = state_batch.clone().detach()#torch.tensor(state_batch,dtype=torch.float32)
        action_batch = action_batch.clone().detach()#torch.tensor(action_batch,dtype=torch.float32)
        reward_batch = reward_batch.clone().detach()#torch.tensor(reward_batch,dtype=torch.float32)
        next_state_batch = next_state_batch.clone().detach()#torch.tensor(next_state_batch,dtype=torch.float32)
        terminal_batch = terminal_batch.clone().detach()#torch.tensor(terminal_batch,dtype=torch.float32)

#         bdc = np.abs(bd-1)
        with torch.no_grad():
            target_critic_input = torch.cat([next_state_batch, self.actor_target(next_state_batch)],dim=1)#*self.action_space_range
#             Zq = self.critic_target(target_critic_input)
            Zq,Zs1 = self.critic_target(target_critic_input)

            intermediary = self.discount*terminal_batch*Zq
            Yq = reward_batch + intermediary
            
        # Update Critic
        self.critic.zero_grad()
        pred_q,pred_s = self.critic(torch.cat([state_batch,action_batch],dim=1))
#         pred_q = self.critic(torch.cat([state_batch,action_batch],dim=1))

        loss_critic_td = criterion(pred_q,Yq)/self.mean_td_error
        loss_critic_model = criterion(pred_s,Zs1)/self.mean_model_error
#         print(f'TD Error for Critic : {loss_critic_td}')
#         print(f'Model Error for Critic : {loss_critic_model}')
#         loss_critic_td.backward()
        torch.autograd.backward([loss_critic_td,0.5*loss_critic_model])
        self.critic_optim.step()
        ### Normalization

        self.mean_td_error = self.mean_td_error + (loss_critic_td.detach()-self.mean_td_error)/(self.update_count+1)
        self.mean_model_error = self.mean_model_error + (loss_critic_model.detach()-self.mean_model_error)/(self.update_count+1)
        self.update_count =  self.update_count + 1
#         print(f'Means: td = {self.mean_td_error}, model = {self.mean_model_error}')
        
        
#         ## Important
#         self.critic.zero_grad()
         # Update Actor
        self.actor.zero_grad()
        actor_loss = -self.critic(torch.cat([state_batch,self.actor(state_batch)],dim=1))[0].mean()#*self.action_space_range

        actor_loss.backward()             
        self.actor_optim.step()  
        self.soft_update()
    
    def soft_update(self):
        self.critic_target.soft_update(self.critic,0.001)
        self.actor_target.soft_update(self.actor,0.001)


        
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

