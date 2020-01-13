
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
from new_models_multi import (Actor, Critic)
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
    def __init__(self,nb_states,nb_actions,nb_goal,args,her=True):
        


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

        self.actor = Actor(self.nb_states, self.nb_actions, self.nb_goals, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, self.nb_goals, **net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic(self.nb_states, self.nb_actions, self.nb_goals, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, self.nb_goals, **net_cfg)
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

        # 
#         if USE_CUDA: self.cuda()

  
    def select_action(self, s_t, decay_epsilon=True):
        if not self.is_training or np.random.uniform() > 0.2:
            action = to_numpy(
                self.actor(torch.cat([to_tensor(np.array([s_t['observation']])),
                                      to_tensor(np.array([s_t['desired_goal']]))],dim=1))).squeeze(0)
    #         action = to_numpy(self.actor(torch.cat([s_t['observation'],
    #                                   s_t['desired_goal']],dim=1)).squeeze(0)
            action += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
        else:
            action = self.random_action()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon *= self.epsilonDecay
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
            Z = self.critic_target(target_critic_input)
            intermediary = self.discount*terminal_batch*Z

            Y = reward_batch + intermediary
        # Update Critic
        self.critic.zero_grad()
        pred = self.critic(torch.cat([state_batch,action_batch,goal_batch],dim=1))
        loss_critic = criterion(pred,Y)
        loss_critic.backward()
        self.critic_optim.step()

         # Update Actor
        self.actor.zero_grad()
        actor_loss = -self.critic(torch.cat([state_batch,
                                             self.actor(torch.cat([state_batch,goal_batch],dim=1)),goal_batch],dim=1)).mean()
        actor_loss.backward()             
        self.actor_optim.step()  
        self.soft_update()

    
    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def soft_update(self):
        self.critic_target.soft_update(self.critic,0.001)
        self.actor_target.soft_update(self.actor,0.001)



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

