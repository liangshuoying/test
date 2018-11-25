import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F

from running_stat import ObsNorm
from distributions import Categorical, DiagGaussian

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

class MLPPolicyTest(FFPolicy):
    def __init__(self, num_inputs, action_space, H1=64, H2=64, H3=64, obs_norm=0):
        super(MLPPolicy, self).__init__()
        
        # DEBUG BEGIN
        #~ self.obs_filter = ObsNorm((1, num_inputs), clip=5, use=obs_norm)
        self.obs_filter = ObsNorm((1, num_inputs), clip=100, use=obs_norm)
        # DEBUG END
#         self.obs_filter = ObsNorm((1, num_inputs), clip=5)
#         self.obs_filter = ObsNorm((1, num_inputs), clip=12)
        self.action_space = action_space
        
        self.lstm = nn.LSTM(3, 128, batch_first=True)
        self.linear1 = nn.Linear(128, 3)
        
        self.a_fc1 = nn.Linear(6, H1)
        self.a_fc2 = nn.Linear(H1, H2)
        self.a_fc3 = nn.Linear(H2, H3)

        self.v_fc1 = nn.Linear(6, H1)
        self.v_fc2 = nn.Linear(H1, H2)
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

    def forward(self, inputs, debug=0):
        inputs.data = self.obs_filter(inputs.data)
        
        print('inputs', inputs.size())
        
        ship = inputs[:, 0]
        ast = inputs[:, 1:]
        
        print('ast', ast.size())
        xx, _ = self.lstm(ast)
        print('xx', xx.size())
        xx = xx.squeeze(1)
        xx = self.linear1(xx)
        y = F.log_softmax(xx)
        
        print('y', y.size())
        
        ast = ast[:, int(y)%ast.size(1) ]
        
        inputs = torch.cat((ship, ast), 1)
        
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


class MLPPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space, H1=64, H2=64, H3=64, obs_norm=1):
        super(MLPPolicy, self).__init__()
        
        # DEBUG BEGIN
        #~ self.obs_filter = ObsNorm((1, num_inputs), clip=5, use=obs_norm)
        self.obs_filter = ObsNorm((1, num_inputs), clip=100, use=obs_norm)
        # DEBUG END
        self.action_space = action_space
        
        # Target Strategies
        self.lstm = nn.LSTM(num_inputs, 128, batch_first=True)
        
        self.a_fc1 = nn.Linear(128, H1)
        self.a_fc2 = nn.Linear(H1, H2)
        self.a_fc3 = nn.Linear(H2, H3)

        #~ self.v_fc1 = nn.Linear(num_inputs, H1)
        #~ self.v_fc2 = nn.Linear(H1, H2)
        #~ self.v_fc3 = nn.Linear(H2, H3)
        self.v_fc4 = nn.Linear(H3, 1)

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

    def forward(self, inputs, debug=0):
        inputs.data = self.obs_filter(inputs.data)
        if debug:
            print('inputs', inputs.size())
        inputs.data = inputs.data.unsqueeze(1)
        x, _ = self.lstm(inputs)
        if debug:
            print('lstm_out', x.size())
        x = x.squeeze(1)
        if debug:
            print('lstm_out', x.size())
        x = self.a_fc1(x)
        x = F.tanh(x)
        
        x = self.a_fc2(x)
        x = F.tanh(x)
        
        x = self.a_fc3(x)
        x = F.tanh(x)
        
        value= self.v_fc4(x)
        
        return value, x
