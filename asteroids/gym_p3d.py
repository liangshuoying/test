import gym
from gym import spaces
from gym.utils import seeding

import torch
import math
import numpy as np
import copy
import collections as col
import os
import time
import random
import socket
import struct

from PIL import Image


def make_env(env_id, seed, rank, log_dir=None, env_args={}):
    def _thunk():
        # args
        obs_pixel = env_args.get('obs_pixel', 0)
        obs_size = env_args.get('obs_size', 9)
        act_disc = env_args.get('act_disc', 0)
        obs_win = env_args.get('obs_win', 4)
        obs_dtype = env_args.get('obs_dtype', None)
        close = env_args.get('close', 0)
        #
        env = Panda3dEnv(id=rank,
                                    obs_pixel=obs_pixel,
                                    obs_size=obs_size,
                                    act_disc=act_disc,
                                    obs_win=obs_win,
                                    obs_dtype=obs_dtype,
                                    close=close,
                                    )
        env.seed(seed + rank)
        return env
    return _thunk


class Panda3dEnv(object):
    
    def __init__(self, id=0, debug=0, obs_pixel=0, obs_size=9, act_disc=0, obs_win=4, obs_dtype=None, close=0):
        self.obs_pixel = obs_pixel
        self.act_disc = act_disc
        
        self.dtype = obs_dtype
        self.window = obs_win
        
        self.close_remote = close
        
        if self.act_disc:
            self.action_space = spaces.Discrete(5)
            # 0-noop, 1-left, 2-right, 3-accel, 4-fire
            self.action_map = {0:[0,-1,-1], 1:[-1,-1,-1], 2:[1,-1,-1], 3:[0,1,-1], 4:[0,-1,1]}
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(3,))
            #~ self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        
        if self.obs_pixel:
            if obs_dtype is None:
                self.observation_space = spaces.Box(low=0, high=255, shape=(1, 84, 84) )
            else:
                self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1, 84, 84) )
                self.buffer = col.deque([], maxlen=self.window)
        else:
            self.obs_fmt = '{}f'.format(obs_size+2)
            # Multi Targets
            #~ self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6, 3,) )
            # Single Target
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_size,) )
        
        # Gym Spec Begin
        self.seed()
        self._spec = None
        self.metadata = {'render.modes': ['human', 'rgb_array']}
        self.reward_range = (-100.0, 100.0)
        self.repeat_action = 0
        # Gym Spec End
        
        # debug begin
        self.frame_count = 0
        # debug end
        
        #~ self.init_p3d_cpu()
        self.init_realtime( port=(8872+id) )
    
    def init_realtime(self, host='localhost', port=8880):
        """ 初始化网络 """
        self.HOST = host
        self.PORT = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    def init_p3d_cpu(self, p3d_exe='AMS.exe'):
        """ 设定进程使用哪个CPU CORE """
        import psutil
        def find_procs_by_name(name):
            ls = []
            for p in psutil.process_iter(attrs=['name']):
                if p.info['name'] == name:
                    ls.append(p)
            return ls
        exe = find_procs_by_name(p3d_exe)
        if exe:
            p = ams[0]
            psu = psutil.Process(p.pid)
            psu.cpu_affinity([2,3])
    
    def _reset_buffer(self):
        for _ in range(self.window):
            self.buffer.append(self.dtype(84, 84).zero_())
    
    def _dbg_pixel(self, obs):
        """ debug only
        """
        self.frame_count += 1
        if self.frame_count%10:
            return
        obs = np.tile(obs[:,:,np.newaxis], (1,1,3))
        im = Image.fromarray(obs, mode='RGB')
        #~ im = Image.fromarray(state, mode='RGBA')
        im.save('./xscreen/%08d.png'%self.frame_count)
    
    #~ def get_realtime_state(self, action=(0, -1), repeat=0):
    def get_realtime_state(self, action=(0, -1, -1), repeat=0):
        """ 发送action,让游戏步进step,返回游戏状态state.
        reset的处理方法:
        由于游戏结束时候P3D已经自动reset了整个游戏,
        但是游戏内的score并没有被重置,
        所以agent reset环境时候只需发送一个空动作,
        让游戏步进一次,把game score设置成0.
        """
        
        action = struct.pack('I3f', repeat, *action)
        #~ action = struct.pack('I2f', repeat, *action)
        
        # request begin
        self.sock.sendto(action, (self.HOST, self.PORT))
        data = self.sock.recv(8192)
        # request end
        
        game_info = {}
        
        if self.obs_pixel:
            d = struct.unpack('7056s3f', data)
            obs, score, done, ship_heading = d
            obs = np.frombuffer(obs, dtype=np.uint8).reshape((84,84))
            game_info.update(ship_heading=ship_heading)
        else:
            #~ d = struct.unpack('14f', data)
            d = struct.unpack(self.obs_fmt, data)
            
            # Single Target
            obs_i = self.observation_space.shape[0]
            obs = np.array( d[:obs_i] )
            # Multi Target
            #~ obs_i = np.multiply(*self.observation_space.shape)
            #~ obs = np.array( d[:obs_i] ).reshape(self.observation_space.shape)
            
            score, done = d[obs_i:]
        
        return obs, score, done, game_info
    
    def reset(self, ):
        """ 重置游戏,
        发送一个空动作reset游戏score.
        """
        if self.close_remote:
            self.get_realtime_state(repeat=99)
        
        obs, _, done, game_info = self.get_realtime_state()
        # Perform up to 30 random no-ops before starting
        #~ for _ in range(random.randrange(30)):
            #~ obs, _, done, game_info = self.get_realtime_state()
        #~ if done:
            #~ obs, _, done, game_info = self.get_realtime_state()
        
        if self.obs_pixel and self.dtype is not None:
            self._reset_buffer()
            self.buffer.append( self.dtype(obs).div_(255) )
            obs = torch.stack(self.buffer, 0)
        
        return obs
    
    #~ def step(self, action, repeat=0):
    def step(self, action, repeat=0):
        """ @action: (turn, accel, fire) - continuous
            {0:nop, 1:left, 2:right, 3:fire} - discrete
        """
        if self.act_disc:
            action = self.action_map.get(action)
        
        info, done = {}, 0
        
        obs, reward, done, game_info = self.get_realtime_state(action, repeat=repeat)
        
        if self.obs_pixel and self.dtype is not None:
            # debug begin
            #~ self._dbg_pixel(obs)
            # debug end
            self.buffer.append( self.dtype(obs).div_(255) )
            obs = torch.stack(self.buffer, 0)
        
        info.update(game_info)
        
        return obs, float(reward), done, info
    
    # ----- Rainbow Dqn Special -----
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False
    
    def action_space_n(self):
        return self.action_space.n
    
    def close(self):
        pass
    
    # ----- 兼容 Gym 环境的一些函数 -----
    @property
    def spec(self):
        return self._spec

    @property
    def unwrapped(self):
        return self
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def render(self, mode='human', close=False):
        """ show something
        """
        return None



