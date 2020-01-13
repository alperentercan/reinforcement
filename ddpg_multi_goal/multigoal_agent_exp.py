
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from buffer import Replay_buffer
from util import *
# from util import to_tensor
# from util import hard_update
from random_process import OrnsteinUhlenbeckProcess
# from memory import SequentialMemory
# from model import (Actor, Critic)
criterion = nn.MSELoss()
from new_models_multi import (Actor, Critic_Fm)


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
    def __init__(self,nb_states,nb_actions,nb_goal,args,her=True):
        

        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions= nb_actions
        self.nb_goals = nb_goal
        self.her = her
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2, 
            'init_w':args.init_w
        }
        
        n_inputs_critic = nb_states + nb_actions + nb_goal
        n_inputs_actor = nb_states  + nb_goal
        n_outputs_actor = nb_actions
        print('n_inputs_actor :',n_inputs_actor)
        
        self.actor = Actor(self.nb_states, self.nb_actions, nb_goal, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions,nb_goal, **net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic_Fm(self.nb_states, self.nb_actions, nb_goal, **net_cfg)
        self.critic_target = Critic_Fm(self.nb_states, self.nb_actions,nb_goal, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)
        
          
        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        self.buffer = Replay_buffer(args.rmsize,self.nb_states,self.nb_actions,self.nb_goals)
        #Create replay buffer
#         self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
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
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.new_episode = False
        self.is_training = True
        self.mean_td_error = 1
        self.mean_model_error = 1
        self.update_count = 0
        self.model_error_coef = 1
        self.model_error_coef_decay = 1/(50*1000*2)

        # 
#         if USE_CUDA: self.cuda()

  
    def select_action(self, s_t, decay_epsilon=True):
        if not self.is_training or np.random.uniform() > 0.2:
#             print('Not random action')
            action = to_numpy(
                self.actor(torch.cat([to_tensor(np.array([s_t['observation']])),
                                      to_tensor(np.array([s_t['desired_goal']]))],dim=1))).squeeze(0)
    #         action = to_numpy(self.actor(torch.cat([s_t['observation'],
    #                                   s_t['desired_goal']],dim=1)).squeeze(0)
            action += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
        else:
#             print('Random action')
            action = self.random_action()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon *= self.epsilonDecay
        self.a_t = action
        return action
    
#     def observe(self,obs,reward,done):
#         super().observe(obs,reward,done)
#         self.buffer.add_entry(self.obs,self.a,self.reward,self.new_obs,int(done))

    def observe(self,reward,obs,done):
        if self.is_training:    
            if self.new_episode:
                self.new_episode = False
                self.s_t = obs
            else:
                if done:
                    self.new_episode = True
                self.buffer.add_entry(self.s_t['observation'], self.a_t, reward,obs['observation'], done,self.s_t['desired_goal'])
                if self.her:
                    self.buffer.add_entry(self.s_t['observation'], self.a_t, 1, obs['observation'], done,self.s_t['achieved_goal'])
#                 self.memory.append(self.s_t, self.a_t, reward, done)
                self.s_t = obs

    

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
    
        
    ### From DDPG IMPLEMENTATION
    def update_policy(self):
#         # Sample batch
#         state_batch, action_batch, reward_batch, \
#         next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch,goal_batch = self.buffer.sample_split_batch(self.batch_size)


        state_batch = state_batch.clone().detach()
        action_batch = action_batch.clone().detach()
        reward_batch = reward_batch.clone().detach()
        next_state_batch = next_state_batch.clone().detach()
        terminal_batch = terminal_batch.clone().detach()
        goal_batch = goal_batch.clone().detach()
#         bdc = np.abs(bd-1)
        with torch.no_grad():
            target_critic_input = torch.cat([next_state_batch,
                                             self.actor_target(torch.cat([next_state_batch,goal_batch],dim=1)),
                                             goal_batch],dim=1)#*self.action_space_range
            Zq,Zs1 = self.critic_target(target_critic_input)
            intermediary = self.discount*terminal_batch*Zq
    #         print(torch.cat([intermediary,Z],1))

    #         print(Z)
    #         print(intermediary)
            Yq = reward_batch + intermediary
        # Update Critic
        self.critic.zero_grad()
        pred_q,pred_s = self.critic(torch.cat([state_batch,action_batch,goal_batch],dim=1))
#         loss_critic = criterion(predq,Y)
                
        loss_critic_td = criterion(pred_q,Yq)/self.mean_td_error
        loss_critic_model = criterion(pred_s,Zs1)/self.mean_model_error
#         if self.update_count%25 == 0:
#             print(f'Mean TD Error : {self.mean_td_error}, mean Model Error {self.mean_model_error}')
        torch.autograd.backward([loss_critic_td,self.model_error_coef*loss_critic_model])

        self.mean_td_error = self.mean_td_error + (loss_critic_td.detach()-self.mean_td_error)/(self.update_count+1)
        self.mean_model_error = self.mean_model_error + (loss_critic_model.detach()-self.mean_model_error)/(self.update_count+1)
        self.update_count =  self.update_count + 1
        self.model_error_coef -= self.model_error_coef_decay

#         loss_critic.backward()
        self.critic_optim.step()

         # Update Actor
        self.actor.zero_grad()
        actor_loss = -self.critic(torch.cat([state_batch,
                                             self.actor(torch.cat([state_batch,goal_batch],dim=1)),goal_batch],dim=1))[0].mean()
        actor_loss.backward()             
        self.actor_optim.step()  
        self.soft_update()

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
    
    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

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
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

#     def seed(self,s):
#         torch.manual_seed(s)
#         if USE_CUDA:
#             torch.cuda.manual_seed(s)
