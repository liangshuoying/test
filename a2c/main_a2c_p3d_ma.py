import copy
import glob
import os
import time

import gym
from gym import spaces

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from agents.agent_a2c import Agent
from agents.agent_a2c_mt import Agent as AgentX
# from agents.agent_a2c_test import Agent as AgentT

from arguments import get_args
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
#~ from envs import make_env
#~ import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from asteroids import make_env_ma as make_env
#~ from astmaze import make_env

from tensorboardX import SummaryWriter


def make_parallel_envs(conf):
    envs = SubprocVecEnv([
            make_env(conf['env_name'], conf['seed'], i, conf['log_dir'], conf['env_args'])
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
    
    ma_mode = conf['ma_train_mode'] if conf['multi_agent'] else 1
    # debug begin
#     ma_mode = 0
    # debug end
    # always train Agent1 if not eval both
    agent1 = Agent(conf.copy(), observation_space, action_space, train=ma_mode, tsbx=self_tsbx, MODEL_ID=TSBX_ID)
    
    if conf['multi_agent']:
        ma_mode = conf['ma_train_mode']==0
        # use different model in eval_both mode if possible
        model1 = conf.get('trained_model', None)
        model2 = conf.get('trained_model2', model1)
        conf.update(trained_model=model2)
        # Only Train Agent2 in train_both_mode
#         agent2 = AgentT(conf.copy(), observation_space, action_space, train=ma_mode, tsbx=self_tsbx, MODEL_ID=TSBX_ID)
        agent2 = Agent(conf.copy(), observation_space, action_space, train=ma_mode, tsbx=self_tsbx, MODEL_ID=TSBX_ID)
        # AgentX Begin
        confx = conf.copy()
        confx.update(trained_model=None, algo='a2c')
        # Multi Targets Strategies Begin
        obs_space_mt = spaces.Box(low=-1.0, high=1.0, shape=(5,) )
        act_space_mt = spaces.Discrete(5)
#         act_space_mt = action_space
        # Multi Targets Strategies End
        hier_train = conf.get('hier_train', 1)
        modelx = conf.get('trained_modelx', None)
        confx.update(trained_model=modelx)
        agentx = AgentX(confx.copy(), obs_space_mt, act_space_mt, train=hier_train, tsbx=self_tsbx, MODEL_ID=TSBX_ID)
        # AgentX End
    
    t_start = time.time()
    t_fps = t_start
    j_fps = 0
    reward_save_max = 0
    
    if conf['multi_agent']:
        state, state2, obs2, obs2x = envs.reset_ma(restart=1)
        agent2.on_init(state2)
        agentx.on_init(obs2x[1:])
        info = dict(obs2=obs2, obs2x=obs2x)
        done = np.zeros((conf['num_processes'], 1))
    else:
        state = envs.reset()
    
    agent1.on_init(state)
    
    for j in range(num_updates):
        for step in range(conf['num_steps']):
            # action
            cpu_actions1 = agent1.on_action(step)
            # action hack begin
#             print('cpu_actions', cpu_actions.shape)
            if conf['multi_agent']:
                # target select
                if conf['hier_agent'] and 'obs2x' in info:
                    if agentx.continuous==1:
                        cpu_actions2 = agentx.on_action(step, obs_ext=info['obs2'])
                    else:
                        target_select = agentx.on_action(step, done=done)
                        selected = [i[j] for i,j in zip(info['obs2x'][1:], target_select)]
                        selected = np.stack(selected)
                        obs2 = np.hstack( (info['obs2'], selected) )
                        obs2 = torch.from_numpy(obs2).float()
                        cpu_actions2 = agent2.on_action(step, obs_new=obs2)
                # target end
                else:
                    cpu_actions2 = agent2.on_action(step)
                cpu_actions = np.hstack( (cpu_actions1, cpu_actions2) )
#             print('cpu_actions hack', cpu_actions.shape)
            # action hack end
            else:
                cpu_actions = cpu_actions1
            
            state, reward, done, info = envs.step_ma(cpu_actions)
            
            agent1.on_store(step, state, reward, done)
            
            if conf['multi_agent']:
                agent2.on_store(step, info['obs20'], info['score2'], done)
            if conf['hier_agent']:
                agentx.on_store(step, info['obs2x'][1:], info['score2'], done)
        
        a_v_d = agent1.optimize(j)
        
        if conf['multi_agent']:
            a_v_d2 = agent2.optimize(j)
            final_rewards2 = agent2.get_final_rewards()
            # test begin
#             a_v_d = a_v_d2
            # test end
        if conf['hier_agent']:
            a_v_dx = agentx.optimize(j, obs_ext=info['obs2'])
            final_rewardsx = agentx.get_final_rewards()
            action_lossx, value_lossx, dist_entropyx = a_v_dx
            # test begin
#             final_rewards2 = final_rewardsx
            # test end
        
        action_loss, value_loss, dist_entropy = a_v_d
        # ---------- logs begin ----------
        final_rewards = agent1.get_final_rewards()
        
        # Multi Agent Mode 1, agent1-training, agent2-eval
        if conf['multi_agent'] and conf['ma_train_mode']==1 and j>0 and j%conf['ma_sync_step']==0:
            agent2.sync_model(agent1)
            print('ma_train_mode 1 >> agent >> agent2')
        # Ma Mode End
        
        # save begin
        if conf['save_dir'] != "" and j>1000 and j%conf['save_interval']==0:
            if conf['multi_agent']:
                if conf['hier_agent'] or final_rewards.mean()>final_rewards2.mean() or not agent2.train:
                    if agent1.train:
                        reward_save_max = agent1.save_model(t_start, j)
                    if agent2.train and agent1.train:
                        agent2.sync_model(agent1)
                        print('sync >> agent1 >> agent2')
                    if conf['hier_agent'] and agentx.train:
                        agentx.save_model(t_start, j)
                elif agent2.train:
                    reward_save_max = agent2.save_model(t_start, j)
                    if agent1.train:
                        agent1.sync_model(agent2)
                        print('sync >> agent2 >> agent1')
            else:
                reward_save_max = agent1.save_model(t_start, j)
        # save end
        
        if j % conf['log_interval'] == 0 and 1:
            t_now = time.time()
            t_used = t_now - t_start
            fps_dt = t_now - t_fps
            fps_dj = j - j_fps
            j_fps = j
            t_fps = t_now
            fps = fps_dj/fps_dt
            t_used_min, t_used_sec = divmod(t_used, 60)
            agent_mode = ''
            if conf['multi_agent']:
                agent_mode = ', ma_mode %d:%d'%(agent1.train, agent2.train)
            print("{:02.0f}m{:02.0f}s, fps:{:.0f}{}".format(t_used_min, t_used_sec, fps, agent_mode))
            print("Updates {}, num frames {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, (j + 1) * conf['num_processes'] * conf['num_steps'],
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(), -dist_entropy.data[0],
                       value_loss.data[0], action_loss.data[0]))
        
        if conf['vis'] and j % conf['vis_interval'] == 0:
            self_tsbx.add_scalar(TSBX_ID+'/entropy', -dist_entropy.data[0], j)
            self_tsbx.add_scalar(TSBX_ID+'/loss_value', value_loss.data[0], j)
            self_tsbx.add_scalar(TSBX_ID+'/loss_policy', action_loss.data[0], j)
            self_tsbx.add_scalars(TSBX_ID+'/reward', {'mean':final_rewards.mean(), 
                                        'min':final_rewards.min(), 'max':final_rewards.max()}, j)
            if conf['multi_agent']:
                self_tsbx.add_scalars(TSBX_ID+'/reward2', {'mean':final_rewards2.mean(), 
                                        'min':final_rewards2.min(), 'max':final_rewards2.max()}, j)
                if conf['hier_agent']:
                    self_tsbx.add_scalar(TSBX_ID+'/x_entropy', -dist_entropyx.data[0], j)
                    self_tsbx.add_scalar(TSBX_ID+'/x_loss_value', value_lossx.data[0], j)
                    self_tsbx.add_scalar(TSBX_ID+'/x_loss_policy', action_lossx.data[0], j)
        
    # end
    envs.close()
    # nicely termination
# ----------------------------------

if __name__ == "__main__":
    args = get_args()
    
    #~ args.entropy_coef = 0.2
    #~ args.value_loss_coef = 0.5
    #~ args.max_grad_norm = 10.
    
    args.num_frames = 1e8
    #~ args.batch_size = 128
    args.batch_size = 64
    #~ args.batch_size = 32
    #~ args.lr = 1e-4
    args.lr = 7e-4
    #~ args.lr = 7e-5
    #~ args.num_stack = 4
    #~ args.num_stack = 3
    #~ args.num_stack = 2
    args.num_stack = 1
#     args.algo = 'a2c'
    args.algo = 'acktr'
#     args.algo = 'ppo'
    # args.clip_param = 0.1
    # args.max_grad_norm = 40
    # args.ppo_epoch = 2
    #~ args.cuda = True
    args.cuda = False
#     args.num_processes = 32
#     args.num_steps = 4
    #~ args.num_processes = 24
    args.num_processes = 16
    #~ args.num_processes = 8
    args.num_steps = 8
    #~ args.num_processes = 1
    
    # SETUP Environment
    args.env_name = "Panda3d_ast"
    #~ args.env_name = "Panda3d_maze"
    # args.env_name = "LunarLanderContinuous-v2"
    #~ args.env_name = "BipedalWalker-v2"
#     args.env_name = "BipedalWalkerHardcore-v2"
    # args.env_name = "Pendulum-v0"
    args.multi_agent = True
#     args.hack_state = True
    args.render_sub0 = True
    args.save_interval = 1000
    # SETUP END
    
    conf = args.__dict__
    
#     conf['normalize_returns'] = 1
#     conf['ma_sync_step'] = 5000
    conf['ma_sync_step'] = 2000
    conf['hier_agent'] = 0
#     conf['hier_agent'] = 1
    conf['hier_train'] = 1
    # 0-eval both, 1-agent2 eval, 2-train both
    #conf['ma_train_mode'] = 0
    conf['ma_train_mode'] = 1
    #conf['ma_train_mode'] = 2
    
    conf['env_args'] = dict(obs_pixel=0, obs_size=8, act_disc=0, obs_win=4, obs_dtype=None)
    #~ conf['env_args'] = dict(obs_pixel=1, act_disc=0, obs_win=4, obs_dtype=None)
    conf['env_args'].update(close=args.close)
#     conf['env_args'].update(close=1)
    
    
    conf['trained_model'] = "acktr_20180409_223026//EP3000_0h8m_160"
    conf['trained_model'] = "acktr_20180409_224147//EP6000_0h15m_120"
    
#     conf['trained_model'] = "acktr_20180308_113147//EP135000_8h47m_360"
#     conf['trained_model'] = "acktr_20180308_231347//EP4000_0h16m_230"
#     
#     conf['trained_model2'] = "acktr_20180227_032250//EP4000_0h11m_120"
    
#     conf['trained_modelx'] = "a2c_20180308_231347//EP4000_0h16m_330"
    
    
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






