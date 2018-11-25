import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical as CategoricalX

from running_stat import ObsNorm
from distributions_mt import Categorical, DiagGaussian

import cv2 as cv

def save_img(name, idx, img):
    img -= img.min()
    img = img/img.max() * 255
    cv.imwrite('./debug/{}_{:08d}.png'.format(name, idx), img)

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

    def act(self, inputs, volatile=False, deterministic=False):
        value, x = self(inputs, volatile=volatile)
        action = self.dist.sample(x, deterministic=deterministic)
        return value, action

    def evaluate_actions(self, inputs, actions):
        value, x = self(inputs)
        action_log_probs, dist_entropy = self.dist.evaluate_actions(x, actions)
        return value, action_log_probs, dist_entropy
    
    def act_eval(self, inputs, obs_ext, volatile=False, deterministic=False):
        value, x = self(inputs, obs_ext, volatile=volatile)
        action, action_log_probs, dist_entropy = self.dist.sample_mt(x, deterministic=deterministic)
        value = value.unsqueeze(0)
        if deterministic:
            return value, action
        return value, action, action_log_probs, dist_entropy
    
    def fck(self, inputs, volatile=False, deterministic=False, done=None):
        return self(inputs, volatile=volatile, deterministic=deterministic)


class CNNPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space):
        super(CNNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        self.linear1 = nn.Linear(32 * 7 * 7, 512)
        
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
        self.out_count = 0
        
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
    
    def debug_conv(self, name, img, debug=1):
        if debug and self.out_count%10000==0:
            print('img', img.size())
            img = img.cpu().data.numpy()[0][0]
            save_img(name, self.out_count, img)
            #~ print('min:', img.min(), 'max:', img.max())
            #~ i = 0
            #~ for img in img.cpu().data.numpy()[0]:
                #~ save_img(name, self.out_count, img)
                #~ i += 1
        if debug>2:
                self.out_count += 1
    
    def forward(self, inputs, debug=0):
#         print(inputs.size())
        #~ x = self.conv1(inputs / 255.0)
        #~ x = self.conv1(inputs / self.obs_max)
        x = inputs / self.obs_max
        # debug begin
        self.debug_conv('cin', x, debug)
        if debug: debug += 1
        # debug end
        x = self.conv1(x)
        # debug begin
        self.debug_conv('conv1', x, debug)
        if debug: debug += 1
        # debug end
        x = F.relu(x)
        #~ x = F.max_pool2d(x, 2)
        
        x = self.conv2(x)
        # debug begin
        self.debug_conv('conv2', x, debug)
        if debug: debug += 1
        # debug end
        x = F.relu(x)
        #~ x = F.max_pool2d(x, 2)
        
        x = self.conv3(x)
        # debug begin
        self.debug_conv('conv3', x, debug)
        # debug end
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
    def __init__(self, num_inputs, action_space, H1=64, H2=64, H3=64, obs_norm=1):
        super(MLPPolicy, self).__init__()
        
        # DEBUG BEGIN
        self.obs_filter = ObsNorm((1, num_inputs), clip=100, use=obs_norm)
        # DEBUG END
        self.action_space = action_space
        
        self.h0 = [None] *16
        self.lstm = nn.GRU(num_inputs, H1, batch_first=True, bidirectional=False)
        
        self.a_fc3 = nn.Linear(H1, 1)
#         self.a_fc3 = nn.Linear(H1, H3)
        self.v_fc4 = nn.Linear(H1, 1)
        
        
#         self.a_fc1 = nn.Linear(num_inputs, H1)
#         self.a_fc2 = nn.Linear(H1, H2)
#         self.a_fc3 = nn.Linear(H2, H3)
# 
#         self.v_fc1 = nn.Linear(num_inputs, H1)
#         self.v_fc2 = nn.Linear(H1, H2)
#         self.v_fc3 = nn.Linear(H2, H3)
#         self.v_fc4 = nn.Linear(H3, 1)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(H3, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(H3, num_outputs)
        else:
            raise NotImplementedError
        
        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init_mlp)

        """
        tanh_gain = nn.init.calculate_gain('tanh')
        self.a_fc1.weight.data.mul_(tanh_gain)
        self.a_fc2.weight.data.mul_(tanh_gain)
        self.v_fc1.weight.data.mul_(tanh_gain)
        self.v_fc2.weight.data.mul_(tanh_gain)
        """

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def cuda(self, **args):
        super(MLPPolicy, self).cuda(**args)
        self.obs_filter.cuda()

    def cpu(self, **args):
        super(MLPPolicy, self).cpu(**args)
        self.obs_filter.cpu()

    def forward_fc(self, inputs, debug=0):
        """ normal fc """
        inputs.data = self.obs_filter(inputs.data)
        
        x = self.v_fc1(inputs)
        x = F.tanh(x)
        
        x = self.v_fc2(x)
        x = F.tanh(x)
        
        x = self.v_fc3(x)
        x = F.tanh(x)
        
        x = self.v_fc4(x)
        value = x
        
        x = self.a_fc1(inputs)
        x = F.tanh(x)
        
        x = self.a_fc2(x)
        x = F.tanh(x)
        
        x = self.a_fc3(x)
        x = F.tanh(x)
        
        return value, x
    
    def forward_gru(self, inputs, debug=0):
        """ normal lstm """
        inputs.data = self.obs_filter(inputs.data)
#         print('inputs', inputs.size())
        inputs = inputs.unsqueeze(1)
#         print('inputs', inputs.size())
        out, _ = self.lstm(inputs)
#         print('out', out.size())
        out = out[:, -1, :]
        
        x = self.v_fc4(out)
        value = x
        
        x = self.a_fc3(out)
        x = F.tanh(x)
        
        return value, x
    
    def forward(self, inputs, obs_ext=None, volatile=False, deterministic=False, done=None):
        """ lstm multi target select """
#         inputs.data = self.obs_filter(inputs.data)
        value = []
        dist_entropy = []
        log_prob = []
        action = []
        new_h0 = []
        if done is not None:
            masks = [0.0 if done_ else 1.0 for done_ in done]
        # 1 step of 16 process
        for obs in inputs:
            obs = obs.astype(np.float32)
            obs = torch.from_numpy(obs)
            obs = Variable(obs.unsqueeze(0), volatile=volatile)
            
#             h0 = self.h0.pop(0)
#             if done is not None:
#                 h0 *= masks.pop(0)
#             
#             out, hx = self.lstm(obs, h0)
#             
#             if volatile:
#                 new_h0.append(h0)
#             else:
#                 new_h0.append(hx)
            
            out, _ = self.lstm(obs)
#             out = out[:, -1, :]
            v = out[:, -1, :]
#             v = out[0, -1, :]
            v = self.v_fc4(v)
            
            value.append(v)
            
            x = self.a_fc3(out)
            x = x.view(1, -1)
#             x = F.tanh(x)
            probs = F.softmax(x, dim=1)
            _log_probs = F.log_softmax(x, dim=1)
            _dist_entropy = -(_log_probs * probs).sum(-1)
            
            dist_entropy.append(_dist_entropy)
            
            m = CategoricalX(probs)
            if deterministic:
                _action = probs.max(1, keepdim=False)[1]
            else:
                _action = m.sample()
            
            _log_prob = m.log_prob(_action)
            
            log_prob.append(_log_prob)
            action.append(_action)
        
        self.h0 = new_h0
        
        value = torch.cat(value).unsqueeze(0)
        dist_entropy = torch.cat(dist_entropy).unsqueeze(1).unsqueeze(0)
        log_prob = torch.cat(log_prob).unsqueeze(1).unsqueeze(0)
        action = torch.cat(action).unsqueeze(1)
        
        # enjoy
        if deterministic:
            return value, action
        # next value
        if volatile:
            return value
        
        return value, action, log_prob, dist_entropy


class MLPPolicyCont(FFPolicy):
    def __init__(self, num_inputs, action_space, H1=64, H2=64, H3=64, obs_norm=1, obs_ext=0):
        super(MLPPolicyCont, self).__init__()
        
        # DEBUG BEGIN
        self.obs_filter = ObsNorm((1, num_inputs), clip=100, use=obs_norm)
        # DEBUG END
        self.action_space = action_space
        
        self.lstm = nn.GRU(num_inputs, H1, batch_first=True, bidirectional=False)
        
#         self.a_fc3 = nn.Linear(H1, H3)
#         self.v_fc4 = nn.Linear(H1, 1)
        
#         self.obs_fc1 = nn.Linear(num_inputs, H1)
#         self.obs_fc1 = nn.Linear(H1, H1)
        self.ext_fc1 = nn.Linear(obs_ext, H1)
        
        self.o_fc1 = nn.Linear(H1+H1, H1)
#         self.o_fc1 = nn.Linear(H1, H1)
#         self.o_fc1 = nn.Linear(8, H1)
        
#         self.a_fc1 = nn.Linear(8, H1)
        self.a_fc2 = nn.Linear(H1, H2)
        self.a_fc3 = nn.Linear(H2, H3)
# 
#         self.v_fc1 = nn.Linear(8, H1)
        self.v_fc2 = nn.Linear(H1, H2)
        self.v_fc3 = nn.Linear(H2, H3)
        self.v_fc4 = nn.Linear(H3, 1)
        
#         self.bn1 = nn.BatchNorm1d(H1+H1, momentum=0.1, affine=True)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(H3, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(H3, num_outputs)
        else:
            raise NotImplementedError
        
        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init_mlp)

        """
        tanh_gain = nn.init.calculate_gain('tanh')
        self.a_fc1.weight.data.mul_(tanh_gain)
        self.a_fc2.weight.data.mul_(tanh_gain)
        self.v_fc1.weight.data.mul_(tanh_gain)
        self.v_fc2.weight.data.mul_(tanh_gain)
        """

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def cuda(self, **args):
        super(MLPPolicy, self).cuda(**args)
        self.obs_filter.cuda()

    def cpu(self, **args):
        super(MLPPolicy, self).cpu(**args)
        self.obs_filter.cpu()

    def forward_fc(self, inputs, debug=0):
        """ normal fc """
        inputs.data = self.obs_filter(inputs.data)
        
        x = self.v_fc1(inputs)
        x = F.tanh(x)
        
        x = self.v_fc2(x)
        x = F.tanh(x)
        
        x = self.v_fc3(x)
        x = F.tanh(x)
        
        x = self.v_fc4(x)
        value = x
        
        x = self.a_fc1(inputs)
        x = F.tanh(x)
        
        x = self.a_fc2(x)
        x = F.tanh(x)
        
        x = self.a_fc3(x)
        x = F.tanh(x)
        
        return value, x
    
    def forward_gru(self, inputs, debug=0):
        """ normal lstm """
        inputs.data = self.obs_filter(inputs.data)
#         print('inputs', inputs.size())
        inputs = inputs.unsqueeze(1)
#         print('inputs', inputs.size())
        out, _ = self.lstm(inputs)
#         print('out', out.size())
        out = out[:, -1, :]
        
        x = self.v_fc4(out)
        value = x
        
        x = self.a_fc3(out)
        x = F.tanh(x)
        
        return value, x
    
    def forward(self, inputs, obs_ext, volatile=False, deterministic=False):
        """ lstm multi target select """
#         inputs.data = self.obs_filter(inputs.data)
        obs_ext = torch.from_numpy(obs_ext.astype(np.float32))
        obs_ext = Variable(obs_ext, volatile=volatile)
        obs_out = []
        # 1 step of 16 process
        for obs in inputs:
            obs = obs.astype(np.float32)
            obs = torch.from_numpy(obs)
            obs = Variable(obs.unsqueeze(0), volatile=volatile)
            
            out, _ = self.lstm(obs)
             
            o = out[:, -1, :]
             
            obs_out.append(o)
        
        obs_out = torch.cat(obs_out)
        
#         print('obs_out', obs_out.size())
#         print('obs_ext', obs_ext.size())
        
#         obs_out = F.tanh(self.obs_fc1(obs_out))
        obs_ext = F.tanh(self.ext_fc1(obs_ext))
        
        obs_out = torch.cat((obs_ext, obs_out), dim=1)
#         obs_out = self.bn1(obs_out)
        
        obs_out = self.o_fc1(obs_out)
        obs_out = F.tanh(obs_out)
        
        x = obs_out
#         x = self.v_fc1(x)
#         x = F.tanh(x)
        
        x = self.v_fc2(x)
        x = F.tanh(x)
        
        x = self.v_fc3(x)
        x = F.tanh(x)
        
        x = self.v_fc4(x)
        value = x
        
        x = obs_out
#         x = self.a_fc1(x)
#         x = F.tanh(x)
        
        x = self.a_fc2(x)
        x = F.tanh(x)
        
        x = self.a_fc3(x)
        action = F.tanh(x)
        
#         print('forward-value', value.size())
#         print('forward-action', action.size())
        
        # enjoy
        if deterministic:
            return value, action
        # next value
        if volatile:
            return value
        
        return value, action





