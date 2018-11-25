import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from running_stat import ObsNorm
from distributions import Categorical, DiagGaussian


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        value, x = self(inputs)
        action = self.dist.sample(x, deterministic=deterministic)
        return value, action

    def evaluate_actions(self, inputs, actions):
        value, x = self(inputs)
        action_log_probs, dist_entropy = self.dist.evaluate_actions(x, actions)
        return value, action_log_probs, dist_entropy


class CNNPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space):
        super(CNNPolicy, self).__init__()
        #~ self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        #~ self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        #~ self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
#         self.linear1 = nn.Linear(32 * 7 * 7, 512)
        #~ self.linear1 = nn.Linear(2048, 512)
        
        #~ self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=1)
        #~ self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        #~ self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        #~ self.linear1 = nn.Linear(64 * 4 * 4, 512)
        # CNN for mines game 1*16*16
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
#         self.linear1 = nn.Linear(1024, 512)    # 1*16*16
        self.linear1 = nn.Linear(1024, 512)     # 3*16*16
        # mines End
        
        self.critic_linear = nn.Linear(512, 1)
        
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(512, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(512, num_outputs)
        else:
            raise NotImplementedError
        
        self.obs_max = 255.
        
        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)
    
    def set_obs_max(self, obs_max=255.):
        self.obs_max = obs_max
    
    def forward(self, inputs):
#         print(inputs.size())
        #~ x = self.conv1(inputs / 255.0)
        x = self.conv1(inputs / self.obs_max)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv3(x)
        x = F.relu(x)
        #~ x = F.max_pool2d(x, 2)
#         print(x.size())
#         x = x.view(-1, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
#         print(x.size())
        x = self.linear1(x)
        x = F.relu(x)

        return self.critic_linear(x), x


def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class MLPPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space, H1=64, H2=64, H3=64, obs_norm=1, use_lstm=0):
        super(MLPPolicy, self).__init__()
        # LSTM BEGIN
        self.use_lstm = use_lstm
        if use_lstm:
            self.hx_v = Variable(torch.zeros(1, H3))
            self.cx_v = Variable(torch.zeros(1, H3))
            self.hx_a = Variable(torch.zeros(1, H3))
            self.cx_a = Variable(torch.zeros(1, H3))
        # LSTM END
        # DEBUG BEGIN
        self.obs_filter = ObsNorm((1, num_inputs), clip=5, use=obs_norm)
        # DEBUG END
#         self.obs_filter = ObsNorm((1, num_inputs), clip=5)
#         self.obs_filter = ObsNorm((1, num_inputs), clip=12)
        self.action_space = action_space
        
        self.a_fc1 = nn.Linear(num_inputs, H1)
        self.a_fc2 = nn.Linear(H1, H2)
        if use_lstm:
            self.a_lstm = nn.LSTMCell(H2, H3)
        else:
            self.a_fc3 = nn.Linear(H2, H3)
        
        self.v_fc1 = nn.Linear(num_inputs, H1)
        self.v_fc2 = nn.Linear(H1, H2)
        if use_lstm:
            self.v_lstm = nn.LSTMCell(H2, H3)
        else:
            self.v_fc3 = nn.Linear(H2, H3)
        self.v_fc4 = nn.Linear(H3, 1)
        
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(H3, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(H3, num_outputs)
        else:
            raise NotImplementedError
        
        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        self.apply(weights_init_mlp)

        """
        tanh_gain = nn.init.calculate_gain('tanh')
        self.a_fc1.weight.data.mul_(tanh_gain)
        self.a_fc2.weight.data.mul_(tanh_gain)
        self.v_fc1.weight.data.mul_(tanh_gain)
        self.v_fc2.weight.data.mul_(tanh_gain)
        """
        if self.use_lstm:
            self.a_lstm.bias_ih.data.fill_(0)
            self.a_lstm.bias_hh.data.fill_(0)
            self.v_lstm.bias_ih.data.fill_(0)
            self.v_lstm.bias_hh.data.fill_(0)
        
        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)
    
    def cuda(self, **args):
        super(MLPPolicy, self).cuda(**args)
        self.obs_filter.cuda()

    def cpu(self, **args):
        super(MLPPolicy, self).cpu(**args)
        self.obs_filter.cpu()

    def forward(self, inputs):
        inputs.data = self.obs_filter(inputs.data)
        
        x = self.v_fc1(inputs)
        x = F.tanh(x)
        
        x = self.v_fc2(x)
        x = F.tanh(x)
        
        if self.use_lstm:
            hx, cx = self.v_lstm(x, (self.hx_v, self.cx_v))
            x = F.tanh(hx)
            self.hx_v = Variable(hx.data)
            self.cx_v = Variable(cx.data)
        else:
            x = self.v_fc3(x)
            x = F.tanh(x)
        
        x = self.v_fc4(x)
        value = x
        
        x = self.a_fc1(inputs)
        x = F.tanh(x)
        
        x = self.a_fc2(x)
        x = F.tanh(x)
        
        if self.use_lstm:
            hx, cx = self.a_lstm(x, (self.hx_a, self.cx_a))
            x = F.tanh(hx)
            self.hx_a = Variable(hx.data)
            self.cx_a = Variable(cx.data)
        else:
            x = self.a_fc3(x)
            x = F.tanh(x)
        
        return value, x



