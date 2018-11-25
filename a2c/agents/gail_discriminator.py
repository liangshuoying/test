import pickle
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from torch.autograd import Variable
from torch import nn

from models.discriminator_mlp import Discriminator
from buffer_gail import ExpertBuffer

class GailAgent(object):
    
    def __init__(self, state_dim, action_dim, buffer_size=1e5, lr=3e-4, 
                                MODEL_ID='Gail', tsbx=None):
        self.MODEL_ID = MODEL_ID
        self.tsbx = tsbx
        
        self.discrim_net = Discriminator(state_dim + action_dim)
        self.optimizer_discrim = torch.optim.Adam(self.discrim_net.parameters(), lr=lr)
        self.discrim_criterion = nn.BCELoss()
        #~ self.expert_traj = ExpertBuffer(buffer_size)
        self.expert_traj = None
        #~ self.load_buffer(buffer_name)
    
    def load_expert(self, name):
        with open(name, 'rb') as f:
            self.expert_traj = pickle.load(f)
    
    def optimize(self, states_b, actions_b, n=3, log_step=1):
        """update discriminator
        @states_b(Tensor)
        @actions_b(Tensor)
        """
        for _ in range(n):
            g_o = self.discrim_net(Variable(torch.cat([states_b, actions_b], 1)))
            #~ e_o = self.discrim_net(Variable(expert_state_actions_b))
            e_o = self.discrim_net(Variable(self.expert_traj.sample(states_b.size(0))))
            self.optimizer_discrim.zero_grad()
            discrim_loss = self.discrim_criterion(g_o, Variable(torch.ones((states_b.shape[0], 1)))) + \
                                self.discrim_criterion(e_o, Variable(torch.zeros((states_b.shape[0], 1))))
            discrim_loss.backward()
            self.optimizer_discrim.step()
        # DEBUG BEGIN
        if self.tsbx and log_step%100==0:
            self.tsbx.add_scalar(self.MODEL_ID+'/discrim_loss', discrim_loss.data[0], log_step)
        # DEBUG END
    
    def expert_reward(self, state, action):
        """batch version
        @state(np.array)
        @action(np.array)
        return (Tensor)
        """
        state_action = torch.Tensor(np.hstack([state, action]))
        #~ return -math.log(self.discrim_net(Variable(state_action, volatile=True)).data.numpy()[0])
        return -(self.discrim_net(Variable(state_action, volatile=True)).log()).data



