import copy
import glob
import os
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from agents.agent_a2c import Agent

from arguments import get_args
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from envs import make_env

from tensorboardX import SummaryWriter


def make_parallel_envs(conf):
    envs = SubprocVecEnv([
            make_env(conf['env_name'], conf['seed'], i, conf['log_dir'])
            for i in range(conf['num_processes'])
        ], render=conf['render_sub0'])
    return envs


def main(conf):
    """ conf ~= args
    """
    assert conf['algo'] in ['a2c', 'ppo', 'acktr']
    if conf['algo'] == 'ppo':
        assert conf['num_processes'] * conf['num_steps'] % conf['batch_size'] == 0
    
    num_updates = int(conf['num_frames']) // conf['num_steps'] // conf['num_processes']
    
    torch.manual_seed(conf['seed'])
    if conf['cuda']:
        torch.cuda.manual_seed(conf['seed'])
    
    os.environ['OMP_NUM_THREADS'] = '1'
    
    MODEL_ID = conf['algo'].upper()
    tsbx_flag = str.replace('%g'%conf['lr'], '.', 'x')
    TSBX_ID = '{}_{}'.format(MODEL_ID, tsbx_flag)
    # -------------------------
    self_tsbx = SummaryWriter()
    
    envs = make_parallel_envs(conf)
    
    observation_space = envs.observation_space
    action_space = envs.action_space
    
    # hack space begin
    if conf['multi_agent']:
        assert action_space.__class__.__name__ == "Box"
        size_to_hack = action_space.shape[0] // 2
        high_ = np.array([np.inf]*24+[1]*size_to_hack)
        # -->> still need to fix action space manually <<--
        action_space = gym.spaces.Box(np.array([-1,-1]), np.array([+1,+1]))
        # manually fix end
        if conf['hack_state']:
            observation_space = gym.spaces.Box(-high_, high_)
    # hack end
    
    agent1 = Agent(conf, observation_space, action_space, 
                                                tsbx=self_tsbx, MODEL_ID=TSBX_ID)
    
    if conf['multi_agent']:
        agent2 =  Agent(conf, observation_space, action_space, 
                                                tsbx=self_tsbx, MODEL_ID=TSBX_ID)
    
    t_start = time.time()
    t_fps = t_start
    j_fps = 0
    reward_save_max = 0
    
    state = envs.reset()
    # state hack begin
#     print('initial state', state.shape)
    if conf['multi_agent']:
        if conf['hack_state']:
            state = np.hstack( (state, np.zeros((conf['num_processes'], size_to_hack)) ) )
        agent2.on_init(state)
    # state hack end
    agent1.on_init(state)
    
    for j in range(num_updates):
        for step in range(conf['num_steps']):
            # action
            cpu_actions1 = agent1.on_action(step)
            # action hack begin
#             print('cpu_actions', cpu_actions.shape)
            if conf['multi_agent']:
                cpu_actions2 = agent2.on_action(step)
                # debug begin by const output only
#                 cpu_actions2 = np.zeros((conf['num_processes'], size_to_hack))
#                 cpu_actions2 = np.ones((conf['num_processes'], size_to_hack))
                # debug end
                cpu_actions = np.hstack( (cpu_actions1, cpu_actions2) )
#             print('cpu_actions hack', cpu_actions.shape)
            # action hack end
            else:
                cpu_actions = cpu_actions1
            
            state, reward, done, info = envs.step(cpu_actions)
            # state hack begin
#             print('running state', state.shape)
            if conf['multi_agent']:
                if conf['hack_state']:
                    state1 = np.hstack( (state, cpu_actions2) )
                    state2 = np.hstack( (state, cpu_actions1) )
                else:
                    state1 = state2 = state
            else:
                state1 = state
#                 print('running state hack', state2.shape)
            # state hack end
            agent1.on_store(step, state1, reward, done)
            if conf['multi_agent']:
                agent2.on_store(step, state2, reward, done)
        
        a_v_d = agent1.optimize(j)
        if conf['multi_agent']:
            a_v_d2 = agent2.optimize(j)
        
        action_loss, value_loss, dist_entropy = a_v_d
        # ---------- logs begin ----------
        final_rewards = agent1.get_final_rewards()
        
        # save begin
        if conf['save_dir'] != "" and (j % conf['save_interval'] == 0 or \
                    (final_rewards.mean()>200 or final_rewards.max()>290) and\
                     final_rewards.max()>reward_save_max):
            reward_save_max = agent1.save_model(t_start, j)
        # save end
        
        if j % conf['log_interval'] == 0:
            t_now = time.time()
            t_used = t_now - t_start
            fps_dt = t_now - t_fps
            fps_dj = j - j_fps
            j_fps = j
            t_fps = t_now
            fps = fps_dj/fps_dt
            t_used_min, t_used_sec = divmod(t_used, 60)
            print("{:02.0f}m{:02.0f}s, fps:{:.0f}".format(t_used_min, t_used_sec, fps))
            print("Updates {}, num frames {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, (j + 1) * conf['num_processes'] * conf['num_steps'],
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(), -dist_entropy.data[0],
                       value_loss.data[0], action_loss.data[0]))
        
        if conf['vis'] and j % conf['vis_interval'] == 0:
            self_tsbx.add_scalar(TSBX_ID+'/entropy', -dist_entropy.data[0], j)
            self_tsbx.add_scalar(TSBX_ID+'/value_loss', value_loss.data[0], j)
            self_tsbx.add_scalar(TSBX_ID+'/policy_loss', action_loss.data[0], j)
            self_tsbx.add_scalars(TSBX_ID+'/reward', {'mean':final_rewards.mean(), 
                                        'min':final_rewards.min(), 'max':final_rewards.max()}, j)
        
    # end
    envs.close()
    # nicely termination
# ----------------------------------

if __name__ == "__main__":
    args = get_args()
    
    args.num_frames = 1e8
    args.batch_size = 128
    # args.lr = 1e-3
    # args.num_stack = 4
    args.algo = 'acktr'
    # args.algo = 'ppo'
    # args.clip_param = 0.1
    # args.max_grad_norm = 40
    # args.ppo_epoch = 2
#     args.cuda = True
    args.cuda = False
#     args.num_processes = 32
#     args.num_steps = 4
#     args.num_processes = 24
    args.num_processes = 16
    args.num_steps = 8
#     args.num_processes = 1
    
    # SETUP Environment
    # args.env_name = "LunarLanderContinuous-v2"
    args.env_name = "BipedalWalker-v2"
#     args.env_name = "BipedalWalkerHardcore-v2"
    # args.env_name = "Pendulum-v0"
#     args.multi_agent = True
#     args.hack_state = True
    args.render_sub0 = True
    # SETUP END
    
    conf = args.__dict__
    
    t_begin = time.time()
    
    main(conf)
    
    t_end = time.time()
    t_used = time.time()-t_begin
    t_used_min, t_used_sec = divmod(t_used, 60)
    t_used_hour, t_used_min = divmod(t_used_min, 60)
    t_begin = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(t_begin) )
    t_end = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(t_end) )
    t_used = "{:02.0f}:{:02.0f}:{:02.0f}".format(t_used_hour, t_used_min, t_used_sec)
    with open('./multi_agent.log', 'a+') as f:
        f.write('Start: {}\n'.format(t_begin) )
        f.write("env: {}\n".format(args.env_name) )
        f.write("algo: {}\n".format(args.algo) )
        f.write('duration: {}\n'.format(t_used) )
        f.write('End: {}\n\n'.format(t_end) )
    
    print("\n----- Finished in {} -----\n".format(t_used) )
    





