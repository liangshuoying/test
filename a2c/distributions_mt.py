import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import AddBias


class Categorical(object):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        x = self.linear(x)
        return x

    def sample(self, x, deterministic):
        x = self(x)

        probs = F.softmax(x, dim=1)
        if deterministic is False:
            action = probs.multinomial()
        else:
            action = probs.max(1, keepdim=True)[1]
        return action

    def evaluate_actions(self, x, actions):
        x = self(x)

        log_probs = F.log_softmax(x, dim=1)
        probs = F.softmax(x, dim=1)

        action_log_probs = log_probs.gather(1, actions)

        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, dist_entropy


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()
        self.fc_mean = nn.Linear(num_inputs, num_outputs)
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        x = self.fc_mean(x)
        action_mean = x

        #  An ugly hack for my KFAC implementation.
        zeros = Variable(torch.zeros(x.size()), volatile=x.volatile)
        if x.is_cuda:
            zeros = zeros.cuda()

        x = self.logstd(zeros)
        action_logstd = x
        return action_mean, action_logstd

    def sample(self, x, deterministic):
        action_mean, action_logstd = self(x)

        action_std = action_logstd.exp()

        noise = Variable(torch.randn(action_std.size()))
        if action_std.is_cuda:
            noise = noise.cuda()

        if deterministic is False:
            action = action_mean + action_std * noise
        else:
            action = action_mean
        return action
    
    def reparameterize(self, mu, logvar, deterministic=False):
        if not deterministic:
#             std = logvar.mul(0.5).exp_()
            std = logvar.mul(1.0).exp_()
#             std = logvar
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
            #~ return eps.mul(std).add_(mu), std
        else:
            return mu
    
    def sample_mt(self, x, deterministic):
        action_mean, action_logstd = self(x)
        
        if 1:
            action_std = action_logstd.exp()
    
            noise = Variable(torch.randn(action_std.size()))
            if action_std.is_cuda:
                noise = noise.cuda()
    
            #~ if deterministic is False:
                #~ action = action_mean + action_std * noise
            #~ else:
                #~ action = action_mean
            
            if deterministic is False:
                action = action_mean.detach() + action_std.detach() * noise
            else:
                action = action_mean.detach()
        else:
            action_std = action_logstd.exp()
            #~ action = self.reparameterize(action_mean, action_logstd, deterministic)
            action = self.reparameterize(action_mean.detach(), action_logstd.detach(), deterministic)
            #~ action, action_std = self.reparameterize(action_mean, action_logstd, deterministic)
            #~ action = action.detach()
            #~ action_mean = action_mean.detach()
        # test begin
        #~ eval_action = Variable(action.data)
        #~ eval_x = Variable(x.data)
        
        #~ action_log_probs, dist_entropy = self.evaluate_actions(eval_x, eval_action)
        
        #~ action = action.unsqueeze(0)
        #~ action_log_probs = action_log_probs.unsqueeze(0)
        #~ dist_entropy = dist_entropy.unsqueeze(0)
        
        #~ return action, action_log_probs, dist_entropy
        # test end
        
        # debug begin
        #~ print('sample_mt: action_mean', action_mean.size())
        #~ print('sample_mt: action_std', action_std.size())
        #~ print('sample_mt: actions', action.size())
        # debug end
        
        action_log_probs = -0.5 * ((action - action_mean) / action_std).pow(2) - 0.5 * math.log(2 * math.pi) - action_logstd
        action_log_probs = action_log_probs.sum(1, keepdim=True)
        dist_entropy = 0.5 + math.log(2 * math.pi) + action_log_probs
        dist_entropy = dist_entropy.sum(-1).mean()
        #~ dist_entropy = dist_entropy.sum(-1)
        
        action = action.unsqueeze(0)
        action_log_probs = action_log_probs.unsqueeze(0)
        dist_entropy = dist_entropy.unsqueeze(0)
        
        # debug begin
        #~ print('sample_mt: action', action.size())
        #~ print('sample_mt: action_log_probs', action_log_probs.size())
        #~ print('sample_mt: dist_entropy', dist_entropy.size())
        # debug end
        
        return action, action_log_probs, dist_entropy
    
    def evaluate_actions(self, x, actions):
        # debug begin
        #~ print('evaluate_actions: x', x.size())
        #~ print('evaluate_actions: actions', actions.size())
        # debug end
        action_mean, action_logstd = self(x)

        action_std = action_logstd.exp()

        action_log_probs = -0.5 * ((actions - action_mean) / action_std).pow(2) - 0.5 * math.log(2 * math.pi) - action_logstd
        action_log_probs = action_log_probs.sum(1, keepdim=True)
        dist_entropy = 0.5 + math.log(2 * math.pi) + action_log_probs
        dist_entropy = dist_entropy.sum(-1).mean()
        # debug begin
        #~ print('evaluate_actions: action_log_probs', action_log_probs.size())
        #~ print('evaluate_actions: dist_entropy', dist_entropy.size())
        # debug end
        return action_log_probs, dist_entropy
