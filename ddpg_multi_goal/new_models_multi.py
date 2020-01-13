
import torch 
import numpy as np

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(torch.nn.Module):
    def __init__(self, nb_states, nb_actions, nb_goal,hidden1=400, hidden2=300, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(nb_states + nb_goal, hidden1)
        self.fc2 = torch.nn.Linear(hidden1, hidden2)
        self.fc3 = torch.nn.Linear(hidden2, nb_actions)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out
    
    def soft_update(self, source, tau):
        for param, source_param in zip(self.parameters(), source.parameters()):
            param.data.copy_(
                param.data * (1.0 - tau) + source_param.data * tau
            )
    

class Critic_Fm(torch.nn.Module):
    def __init__(self,nb_state,nb_action, nb_goal, hidden1=400, hidden2=300, init_w=3e-3):
        ci  = nb_state + nb_action + nb_goal
        super(Critic_Fm,self).__init__()
        self.hl1 = torch.nn.Linear(ci,hidden1)
        self.ha1 = torch.nn.ReLU()
        self.hl2 = torch.nn.Linear(hidden1,hidden2)
        self.ha2 = torch.nn.ReLU()
#         self.hl3 = torch.nn.Linear(200,100)
#         self.ha3 = torch.nn.ReLU()
        self.olv = torch.nn.Linear(hidden2,1)
#         self.ova = torch.nn.Tanh()
        self.ols = torch.nn.Linear(hidden2,nb_state)
        self.init_weights(init_w)

   
    def init_weights(self, init_w):
        self.hl1.weight.data = fanin_init(self.hl1.weight.data.size())
        self.hl2.weight.data = fanin_init(self.hl2.weight.data.size())
        self.olv.weight.data.uniform_(-init_w, init_w)
        self.ols.weight.data.uniform_(-init_w, init_w)

    def forward(self,x):
        
        z1_ = self.hl1(x)
        z1  = self.ha1(z1_)       
        z2_ = self.hl2(z1)
        z2  = self.ha2(z2_)
#         z3_ = self.hl3(z2)
#         z3  = self.ha3(z3_)
        q  = self.olv(z2)
#         q   = self.ova(q_)
        s   = self.ols(z2)
        return q,s
    
    
    def soft_update(self, source, tau):
        for param, source_param in zip(self.parameters(), source.parameters()):
            param.data.copy_(
                param.data * (1.0 - tau) + source_param.data * tau
            )
        
        
        
class Critic(torch.nn.Module):
    def __init__(self, nb_states, nb_actions,nb_goals, hidden1=400, hidden2=300, init_w=3e-3):
        super(Critic, self).__init__()
#         self.fc1 = nn.Linear(nb_states, hidden1)
#         self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
        self.fc1 = torch.nn.Linear(nb_states+nb_actions + nb_goals, hidden1)
        self.fc2 = torch.nn.Linear(hidden1, hidden2)
        self.fc3 = torch.nn.Linear(hidden2, 1)
        self.relu = torch.nn.ReLU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, xs):
#         x, a = xs
#         out = self.fc1(x)
#         out = self.fc1(torch.cat(xs,1))
        out = self.fc1(xs)
        out = self.relu(out)
        # debug()
#         out = self.fc2(torch.cat([out,a],1))
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
    def soft_update(self, source, tau):
        for param, source_param in zip(self.parameters(), source.parameters()):
            param.data.copy_(
                param.data * (1.0 - tau) + source_param.data * tau
            )        

    
