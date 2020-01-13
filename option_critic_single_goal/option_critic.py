import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from buffer_option import Replay_buffer
from util import *
# from util import to_tensor
# from util import hard_update
from random_process import OrnsteinUhlenbeckProcess
# from memory import SequentialMemory
# from model import (Actor, Critic)
criterion = nn.MSELoss()


class Option_critic(object):
    def __init__(self,nb_states,nb_actions,nb_options,args):
        
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.nb_options = nb_options
        
        ## Network Lookups
        nb_actor_in = nb_states + 1
        nb_actor_out = nb_actions
        nb_term_in = nb_states + 1
        nb_term_out = 1
        nb_qintra_in = nb_states + 1 + nb_actions ## qintra is Qu from the paper
        nb_qintra_out = 1
        
        ## Initialize Networks
        
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2, 
            'init_w':args.init_w
        }
        
        ## Termination
        self.terminate = torch.nn.Sequential(
        torch.nn.Linear(nb_term_in, args.hidden1),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden1, args.hidden2),
        torch.nn.ReLU(),       
        torch.nn.Linear(args.hidden2, nb_term_out),
        torch.nn.Sigmoid())
        
        ## Actor
        self.actor = torch.nn.Sequential(
        torch.nn.Linear(nb_actor_in, args.hidden1),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden1,args.hidden2),
        torch.nn.ReLU(),       
        torch.nn.Linear(args.hidden2, nb_actor_out),
        torch.nn.Tanh())

        ## Q-Intra      
        self.qintra = torch.nn.Sequential(
        torch.nn.Linear(nb_qintra_in, args.hidden1),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden1,args.hidden2),
        torch.nn.ReLU(),       
        torch.nn.Linear(args.hidden2, nb_qintra_out))
#         torch.nn.Tanh()) 
        
        ### Target Networks
        ## Termination
        self.target_terminate = torch.nn.Sequential(
        torch.nn.Linear(nb_term_in, args.hidden1),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden1, args.hidden2),
        torch.nn.ReLU(),       
        torch.nn.Linear(args.hidden2, nb_term_out),
        torch.nn.Sigmoid())
        
        ## Actor
        self.target_actor = torch.nn.Sequential(
        torch.nn.Linear(nb_actor_in, args.hidden1),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden1,args.hidden2),
        torch.nn.ReLU(),       
        torch.nn.Linear(args.hidden2, nb_actor_out),
        torch.nn.Tanh())

        ## Q-Intra      
        self.target_qintra = torch.nn.Sequential(
        torch.nn.Linear(nb_qintra_in, args.hidden1),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden1,args.hidden2),
        torch.nn.ReLU(),       
        torch.nn.Linear(args.hidden2, nb_qintra_out))
#         torch.nn.Tanh()) 
        
        

        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)
        self.terminate_optim  = Adam(self.terminate.parameters(), lr=args.rate)
        self.qintra_optim  = Adam(self.qintra.parameters(), lr=args.rate)

        hard_update(self.target_actor, self.actor) # Make sure target is with the same weight
        hard_update(self.target_qintra, self.qintra)
        hard_update(self.target_terminate, self.terminate)

        self.buffer = Replay_buffer(args.rmsize,self.nb_states,self.nb_actions)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        # 
        finalEpsilon = 0.01   # value of epsilon at end of simulation. Decay rate is calculated
        self.epsilonDecay =  np.exp(np.log(finalEpsilon) / (args.epsilon)) # to produce this final value
        self.epsilon = 1.0
        self.epsilon_option = 0.2
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.w_t = None # Most recent option
        self.new_episode = False
        self.is_training = True

        
        
        
        
    def update_policy(self):
#         # Sample batch

        state_batch, action_batch,option_batch,reward_batch, \
        next_state_batch, terminal_batch = self.buffer.sample_split_batch(self.batch_size)
        


        state_batch = state_batch.clone().detach()
        action_batch = action_batch.clone().detach()
        option_batch = option_batch.clone().detach()
        reward_batch = reward_batch.clone().detach().reshape(-1,1)
        next_state_batch = next_state_batch.clone().detach()
        terminal_batch = terminal_batch.clone().detach().reshape(-1,1)
#         goal_batch = goal_batch.clone().detach()

        with torch.no_grad():  
            target_terminate_output_onw = self.target_terminate(torch.cat([next_state_batch,option_batch],dim=1))
            qintra_values = np.zeros((self.batch_size,self.nb_options))
            for i in range(self.nb_options):
                qintra_values[:,i:i+1] =(self.target_qintra(torch.cat([
                    next_state_batch,
                    i*torch.ones((self.batch_size,1)),
                    self.target_actor(torch.cat([next_state_batch,i*torch.ones((self.batch_size,1))],dim=1))],dim=1)))



            target_qintra_overw = torch.tensor(np.amax(qintra_values,axis=1),dtype=torch.float).reshape(-1,1)
#             _,target_qintra_overw = torch.max(qintra_values,axis=1)

#             Q =   self.target_qintra(torch.cat([
#                                             next_state_batch,
#                                             option_batch,
#                                             self.target_actor(torch.cat([next_state_batch,
#                                                                          option_batch],dim=1))],dim=1))

            
            Y =  (reward_batch + terminal_batch*self.discount*((1-target_terminate_output_onw) * 
                                             self.target_qintra(torch.cat([next_state_batch,
                                                                            option_batch,
                                                                            self.target_actor(torch.cat([next_state_batch,
                                                                                                             option_batch],dim=1))],dim=1)))
                            +terminal_batch*self.discount*(target_terminate_output_onw)*target_qintra_overw)
            
        # Update Qintra
        self.qintra.zero_grad()
        pred = self.qintra(torch.cat([state_batch,option_batch,action_batch],dim=1))
        loss_qintra = criterion(pred,Y)
        loss_qintra.backward()
        self.qintra_optim.step()

        # Update Actor
        self.actor.zero_grad()
        actor_loss = -self.qintra(torch.cat([state_batch,
                                             option_batch,
                                             self.actor(torch.cat([state_batch,option_batch],dim=1))],dim=1)).mean()
        actor_loss.backward()             
        self.actor_optim.step()  

        # Update Terminate
        self.terminate.zero_grad()
#         self.qintra.zero_grad()
#         self.actor.zero_grad()
        ## Not advantage
        advantage_batch = self.qintra(torch.cat([next_state_batch,
                                                 option_batch,
                                                 self.actor(torch.cat([next_state_batch,option_batch],dim=1))],dim=1))
        terminate_loss = (self.terminate(torch.cat([next_state_batch,option_batch],dim=1))*advantage_batch).mean()
        terminate_loss.backward()
        self.terminate_optim.step()
        self.soft_update()
    
  
    def select_action(self, s_t, decay_epsilon=True):
        if self.w_t == None:
            if np.random.uniform() > self.epsilon_option:
                option_qs = [self.qintra(torch.cat([to_tensor(np.append(s_t,w)),#s_t,
#                                                    w,
                                                   self.actor(to_tensor(np.append(s_t,w)))])) for w in range(self.nb_options)] 
#                                                    self.actor(torch.cat([s_t,w],dim=1))],dim=1)) for w in range(self.nb_options)] 
                ind = range(self.nb_options)
                self.w_t = max(ind,key=lambda x:option_qs[x])
            else:
                self.w_t = np.random.randint(self.nb_options)
        old_option = self.w_t
        if self.terminate(to_tensor(np.append(s_t,self.w_t))) == 1:
            if np.random.uniform() > self.epsilon_option:

                option_qs = [self.qintra(torch.cat([to_tensor(np.append(s_t,w)),#s_t,
#                                                        w,
                                                   self.actor(to_tensor(np.append(s_t,w)))])) for w in range(self.nb_options)] 
                ind = range(self.nb_options)
                self.w_t = max(ind,key=lambda x:option_qs[x])
            else:
                self.w_t = np.random.randint(self.nb_options)
        action = to_numpy(
            self.actor(to_tensor(np.append(s_t,self.w_t))))#.squeeze(0) 
#         print(action.shape)
#         action = to_numpy(
#             self.actor(torch.cat([to_tensor(np.array([s_t])),
#                                   torch.tensor([self.w_t],dtype=torch.float)],dim=1))).squeeze(0)
        action = action.reshape(1,-1)
        action += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon
#         print(f"State {s_t}, old option was {old_option}, new option is {self.w_t} and the action is {action}")
        self.a_t = action
        return action
    

    def observe(self,reward,obs,done):

        if self.is_training:    
            if self.new_episode:
                self.new_episode = False
                self.s_t = obs
                
            else:
                if done:
                    self.new_episode = True
#                 print(np.array([self.w_t]).shape)
                self.buffer.add_entry(self.s_t, self.a_t,self.w_t,reward,obs, done)
#                 self.memory.append(self.s_t, self.a_t, reward, done)
                self.s_t = obs
    

    def random_action(self):
        action = np.random.uniform(-1.,1.,self.nb_actions)#*self.action_space_range
        self.a_t = action
        return action  

    def soft_update(self):      
        #Update target networks          
        with torch.no_grad():
            for i in [0,2,4]:
                self.target_qintra[i].weight.data = (self.tau*self.qintra[i].weight.data.clone() + 
                                                     (1-self.tau) *self.target_qintra[i].weight.data.clone())
            for i in [0,2,4]:
                self.target_actor[i].weight.data = (self.tau*self.actor[i].weight.data.clone() + 
                                                    (1-self.tau)*self.target_actor[i].weight.data.clone())
                           
            for i in [0,2,4]:
                self.target_terminate[i].weight.data = (self.tau*self.terminate[i].weight.data.clone() + 
                                                    (1-self.tau)*self.target_terminate[i].weight.data.clone())


    def save_model(self,output):
        return 0

    def reset(self, obs):
        self.s_t = obs
        ## initialize option
        if np.random.uniform() > self.epsilon_option:
#             print(obs)
#             print(to_tensor(np.append(obs,1)).shape)
#             print(self.actor(to_tensor(np.append(obs,1))).shape)
#             print(torch.cat([torch.tensor(np.append(obs,1),dtype=torch.float32),self.actor(to_tensor(np.append(obs,1)))]).shape)
#             print(np.append(obs,1).shape)
            option_qs = [self.qintra(torch.cat([to_tensor(np.append(obs,w),dtype=torch.float32),
#                                                torch.tensor(w),
                                               self.actor(to_tensor(np.append(obs,w)))])) for w in range(self.nb_options)]
#             option_qs = [np.append(obs,w) for w in range(self.nb_options)] 
#             print(option_qs)
            ind = range(self.nb_options)
            self.w_t = max(ind,key=lambda x:option_qs[x])
        else:
            self.w_t = np.random.randint(self.nb_options)
        self.random_process.reset_states()
