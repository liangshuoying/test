import pickle
import copy
import glob
import time
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from kfac import KFACOptimizer
from storage_mt import RolloutStorage

from models.model_mlp_gru import CNNPolicy
from models.model_mlp_gru import MLPPolicy
# from models.model_mlp_gru import MLPPolicyCont as MLPPolicy


class Agent(object):
    """ a2c agent
    """
    def __init__(self, conf, observation_space, action_space, train=1, tsbx=None, MODEL_ID='A2C'):
        self.conf = conf
        self.tsbx = tsbx if conf['vis'] else None
        self.envs_observation_space_shape0 = observation_space.shape[0]
        self.train = train
        
#         tsbx_flag = str.replace('%g'%conf['lr'], '.', 'x')
#         self.TSBX_ID = '{}_{}'.format(MODEL_ID, tsbx_flag)
        
        obs_shape = observation_space.shape
        # py3 style
    #     obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
        # py27 style
        obs_shape = (obs_shape[0] * conf['num_stack'],) + tuple(obs_shape[1:])
        
        if len(observation_space.shape) == 3:
            self.actor_critic = CNNPolicy(obs_shape[0], action_space)
        else:
            self.actor_critic = MLPPolicy(obs_shape[0], action_space, H1=64, H2=64)
#             self.actor_critic = MLPPolicy(obs_shape[0], action_space, H1=64, H2=64, obs_ext=3)
#             self.actor_critic = MLPPolicy(obs_shape[0], action_space, H1=128, H2=128)
#             self.actor_critic = MLPPolicy(obs_shape[0], action_space, H1=256, H2=128, H3=64)
        
        if action_space.__class__.__name__ == "Discrete":
            action_shape = 1
            self.continuous = 0
        else:
            action_shape = action_space.shape[0]
            self.continuous = 1
        
        self.action_shape = action_shape
        self.obs_shape = obs_shape
        
        if conf['cuda']:
            self.actor_critic.cuda()
        
        if conf['algo'] == 'a2c':
            self.optimizer = optim.RMSprop(self.actor_critic.parameters(), conf['lr'], eps=conf['eps'], alpha=conf['alpha'])
        elif conf['algo'] == 'ppo':
            self.optimizer = optim.Adam(self.actor_critic.parameters(), conf['lr'], eps=conf['eps'])
        elif conf['algo'] == 'acktr':
            self.optimizer = KFACOptimizer(self.actor_critic)
        
        # load trained
        actor_critic, acktr_optimizer = self.load_model(conf)
        if actor_critic:
            self.actor_critic.obs_filter = copy.deepcopy(actor_critic.obs_filter)
            self.actor_critic.load_state_dict(actor_critic.state_dict())
            print('[*] Loaded: actor_critic')
            del actor_critic
        if acktr_optimizer:
            self.optimizer.load_state_dict(acktr_optimizer.state_dict())
            print('[*] Loaded: acktr_optimizer')
            del acktr_optimizer
        # load end
        
        if not self.train:
            self.actor_critic.eval()
            print('!!! agent in eval mode !!!')
        
        normalize_returns = conf.get('normalize_returns', 0)
        self.rollouts = RolloutStorage(conf['num_steps'], conf['num_processes'], obs_shape, action_space, normalize_returns)
#         self.current_state = torch.zeros(conf['num_processes'], *obs_shape)
        self.current_state = np.zeros(conf['num_processes'], dtype=np.object)
        
#         state = envs.reset()
#         self.update_current_state(state)
#         self.rollouts.states[0].copy_(self.current_state)
    
        # These variables are used to compute average rewards for all processes.
        self.episode_rewards = torch.zeros([conf['num_processes'], 1])
        self.final_rewards = torch.zeros([conf['num_processes'], 1])
        
        if conf['cuda']:
            self.current_state = self.current_state.cuda()
            self.rollouts.cuda()
    
        if conf['algo'] == 'ppo':
            self.old_model = copy.deepcopy(self.actor_critic)
    
    def get_final_rewards(self):
        return self.final_rewards
    
    def update_current_state(self, state):
        self.current_state[:] = state
#         self.current_state = state
    
    def on_init(self, state):
        """state = envs.reset()"""
        self.update_current_state(state)
#         self.rollouts.states[0].copy_(self.current_state)
        self.rollouts.states[0] = self.current_state
        # set current vars
        self.current_value = None
        self.current_action = None
        # set end
        self.value_batch = []
        self.log_prob_batch = []
        self.dist_entropy_batch = []
    
    def on_action(self, step, obs_ext=None, done=None):
        if self.train:
#             value, action = self.actor_critic.act(Variable(self.rollouts.states[step], volatile=True))
            value, action, log_prob, dist_entropy = self.actor_critic.fck(self.rollouts.states[step], done=done)
#             value, action, log_prob, dist_entropy = self.actor_critic.act_eval(self.rollouts.states[step], obs_ext)
            self.value_batch.append(value)
            self.log_prob_batch.append(log_prob)
            self.dist_entropy_batch.append(dist_entropy)
        else:
            value, action = self.actor_critic.fck(self.rollouts.states[step], volatile=True, deterministic=True)
#             value, action = self.actor_critic.act_eval(self.rollouts.states[step], obs_ext, volatile=True, deterministic=True)
#             value, action = self.actor_critic.act(Variable(self.rollouts.states[step], volatile=True), deterministic=True)
        
        # debug begin
#         print('on_action action', action.size())
#         print('on_action value', value.size())
#         print('on_action log_prob', log_prob.size())
#         print('on_action dist_entropy', dist_entropy.size())
        # debug end
#         cpu_actions = action.data.cpu().numpy()
#         cpu_actions = action.data.squeeze(1).cpu().numpy()
        # clip begin, clip continuous only
        if self.continuous:
            cpu_actions = action.data.squeeze(0).cpu().numpy()
            cpu_actions = cpu_actions.clip(-1, 1)
        else:
            cpu_actions = action.data.squeeze(1).cpu().numpy()
        # clip end
        # set current vars
        self.current_value = value.squeeze(0)
        self.current_action = action.squeeze(0)
        # set end
        return cpu_actions
    
    def push_obs(self, step, state):
        self.update_current_state(state)
        self.rollouts.push(step, self.current_state)
    
    def on_store(self, step, state, reward, done):
        reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
        self.episode_rewards += reward
        
        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        self.final_rewards *= masks
        self.final_rewards += (1 - masks) * self.episode_rewards
        self.episode_rewards *= masks
        
        if self.conf['cuda']:
            masks = masks.cuda()
        
#         if self.current_state.dim() == 4:
#         if self.current_state.ndim == 4:
#             self.current_state *= masks.unsqueeze(2).unsqueeze(2)
#         else:
#             self.current_state *= masks.squeeze(1)
#             self.current_state *= masks
        
        self.update_current_state(state)
        self.rollouts.insert(step, self.current_state, self.current_action.data, self.current_value.data, reward, masks)
    
    def optimize(self, j=0, obs_ext=None):
        if not self.train:
            return [Variable(torch.Tensor([0]))] *3
        rollouts = self.rollouts
        actor_critic = self.actor_critic
        optimizer = self.optimizer
        
#         next_value = actor_critic(rollouts.states[-1], volatile=True)[0].data
        next_value = actor_critic(rollouts.states[-1], obs_ext, volatile=True).data
#         next_value = actor_critic(Variable(rollouts.states[-1], volatile=True))[0].data

#         if hasattr(actor_critic, 'obs_filter'):
#             actor_critic.obs_filter.update(rollouts.states[:-1].view(-1, *self.obs_shape))

        rollouts.compute_returns(next_value, self.conf['use_gae'], self.conf['gamma'], self.conf['tau'])
        
        if self.conf['algo'] in ['a2c', 'acktr']:
#             values, action_log_probs, dist_entropy = actor_critic.evaluate_actions(
#                                                         Variable(rollouts.states[:-1].view(-1, *self.obs_shape)), 
#                                                         Variable(rollouts.actions.view(-1, self.action_shape))
#                                                     )
            
#             values = values.view(self.conf['num_steps'], self.conf['num_processes'], 1)
#             action_log_probs = action_log_probs.view(self.conf['num_steps'], self.conf['num_processes'], 1)
            values = torch.cat(self.value_batch)
            action_log_probs = torch.cat(self.log_prob_batch)
            dist_entropy = torch.cat(self.dist_entropy_batch).mean()
            # debug begin
#             print('optimize: values', values.size())
#             print('optimize: action_log_probs', action_log_probs.size())
#             print('optimize: dist_entropy', dist_entropy.size())
#             print('optimize: rollouts.returns[:-1]', rollouts.returns[:-1].shape)
            # debug end
            advantages = Variable(rollouts.returns[:-1]) - values
            value_loss = advantages.pow(2).mean()
            action_loss = -(Variable(advantages.data) * action_log_probs).mean()
            
            if self.conf['algo'] == 'acktr' and optimizer.steps % optimizer.Ts == 0:
                # Sampled fisher, see Martens 2014
                actor_critic.zero_grad()
                pg_fisher_loss = -action_log_probs.mean()

                value_noise = Variable(torch.randn(values.size()))
                if self.conf['cuda']:
                    value_noise = value_noise.cuda()

                sample_values = values + value_noise
                vf_fisher_loss = -(values - Variable(sample_values.data)).pow(2).mean()

                fisher_loss = pg_fisher_loss + vf_fisher_loss
                optimizer.acc_stats = True
                fisher_loss.backward(retain_graph=True)
                optimizer.acc_stats = False

            optimizer.zero_grad()
            (value_loss * self.conf['value_loss_coef'] + action_loss - dist_entropy * self.conf['entropy_coef']).backward()
#             (value_loss * self.conf['value_loss_coef'] + action_loss - dist_entropy * self.conf['entropy_coef']).backward(retain_graph=True)

            if self.conf['algo'] == 'a2c':
                # DEBUG LOG BEGIN
                if self.tsbx and j%100==0 and 1:
                    for name, param in actor_critic.named_parameters():
                        self.tsbx.add_histogram('actor_critic.%s'%name, param.clone().cpu().data.numpy(), j)
#                         self.tsbx.add_histogram('actor_critic.%s_grads'%name, param.grad.clone().data.numpy(), j)
                # DEBUG END
                nn.utils.clip_grad_norm(actor_critic.parameters(), self.conf['max_grad_norm'])

            optimizer.step()
            # clean begin
            del self.value_batch[:]
            del self.log_prob_batch[:]
            del self.dist_entropy_batch[:]
            # clean end
        elif self.conf['algo'] == 'ppo':
            advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            
            old_model = self.old_model
            old_model.load_state_dict(actor_critic.state_dict())
            if hasattr(actor_critic, 'obs_filter'):
                old_model.obs_filter = actor_critic.obs_filter

            for _ in range(self.conf['ppo_epoch']):
                sampler = BatchSampler(SubsetRandomSampler(range(self.conf['num_processes'] * self.conf['num_steps'])), 
                                                self.conf['batch_size'] * self.conf['num_processes'], drop_last=False)
                for indices in sampler:
                    indices = torch.LongTensor(indices)
                    if self.conf['cuda']:
                        indices = indices.cuda()
                    states_batch = rollouts.states[:-1].view(-1, *self.obs_shape)[indices]
                    actions_batch = rollouts.actions.view(-1, self.action_shape)[indices]
                    return_batch = rollouts.returns[:-1].view(-1, 1)[indices]

                    # Reshape to do in a single forward pass for all steps
                    values, action_log_probs, dist_entropy = actor_critic.evaluate_actions(Variable(states_batch), Variable(actions_batch))

                    _, old_action_log_probs, _ = old_model.evaluate_actions(Variable(states_batch, volatile=True), Variable(actions_batch, volatile=True))

                    ratio = torch.exp(action_log_probs - Variable(old_action_log_probs.data))
                    adv_targ = Variable(advantages.view(-1, 1)[indices])
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - self.conf['clip_param'], 1.0 + self.conf['clip_param']) * adv_targ
                    action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)

                    value_loss = (Variable(return_batch) - values).pow(2).mean()

                    optimizer.zero_grad()
                    (value_loss + action_loss - dist_entropy * self.conf['entropy_coef']).backward()
                    optimizer.step()

#         rollouts.states[0].copy_(rollouts.states[-1])
        rollouts.states[0] = rollouts.states[-1]
        
        return action_loss, value_loss, dist_entropy
    
    def save_model(self, t_start, j):
        final_rewards = self.get_final_rewards()
        
        a2c_name = self.conf['algo'] + time.strftime('_%Y%m%d_%H%M%S', time.localtime(t_start))
        ENV_NAME_FIX = self.conf['env_name'].replace('-', '_')
        
#         save_path = os.path.join(self.conf['save_dir'], self.conf['algo'])
        save_path = os.path.join(self.conf['save_dir'], ENV_NAME_FIX, a2c_name)
        try:
            os.makedirs(save_path)
        except OSError:
            pass

        # A really ugly way to save a model to CPU
        save_model = self.actor_critic
        #~ if args.cuda:
            #~ save_model = copy.deepcopy(actor_critic).cpu()
            #~ save_model = actor_critic.cpu()
        print("----- save model -----")
        reward_save_max = final_rewards.max()
        t_save = time.time() - t_start
        t_save = "{:.0f}h{:.0f}m".format(t_save//3600, t_save%3600/60)
#         model_name = "{}_{}_{}_{:.0f}".format(a2c_name, j, t_save, reward_save_max)
        model_name = "EP{}_{}_{:.0f}".format(j, t_save, reward_save_max)
        torch.save(save_model, os.path.join(save_path, model_name + ".mt"))
        #~ torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))
        #~ torch.save(save_model.state_dict(), os.path.join(save_path, args.env_name + ".pt"))
        # save acktr optimizer
        if self.conf['algo'] == 'acktr':
            self.save_acktr(os.path.join(save_path, model_name + ".acktr"))
        # acktr end
        return reward_save_max
    
    def save_acktr(self, acktr_model):
        with open(acktr_model, 'wb') as f:
            pickle.dump(self.optimizer, f)
    
    def load_model(self, conf):
        actor_critic = None
        acktr_optimizer = None
        if conf.get('trained_model', None):
            load_dir = os.path.join(conf['save_dir'], conf['env_name'].replace('-', '_'))
            actor_critic = torch.load(os.path.join(load_dir, conf['trained_model'] + ".mt"))
            if conf['algo'] == 'acktr':
                with open(os.path.join(load_dir, conf['trained_model'] + ".acktr"), 'rb') as f:
                    acktr_optimizer = pickle.load(f)
        return actor_critic, acktr_optimizer
    
    def sync_model(self, model):
        self.actor_critic.load_state_dict(model.actor_critic.state_dict())
        self.actor_critic.obs_filter = copy.deepcopy(model.actor_critic.obs_filter)
    
    def train_mode(self):
        self.actor_critic.train()
        self.train = 1
    
    def eval_mode(self):
        self.actor_critic.eval()
        self.train = 0




