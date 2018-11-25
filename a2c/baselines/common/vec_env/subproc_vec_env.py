import numpy as np
# from multiprocessing import Process, Pipe
from torch.multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv

def worker(remote, env_fn_wrapper, render=0):
    env = env_fn_wrapper.x()
    i = 0
    reward_sum = 0
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
                # debug begin
                #~ if render:
                    #~ print('reward_sum {:.0f}'.format(reward_sum))
                reward_sum = 0
                # debug end
            remote.send((ob, reward, done, info))
            # render begin
            reward_sum += reward
            i += 1
            if render and i%10==0:
                env.render('human')
            # render end
        elif cmd == 'step_ma':
            ob, reward, done, info = env.step(data)
            # DEBUG BEGIN
            #~ if render:
                #~ print('done', done)
            # DEBUG END
            if done:
                ob, obs20, obs2, obs2x = env.reset()
                info.update(obs20=obs20, obs2=obs2, obs2x=obs2x)
                # debug begin
                #~ if render:
                    #~ print(ob)
                    #~ print(obs2)
                    #~ print('-'*10+'reward_sum {:.0f}'.format(reward_sum))
                reward_sum = 0
                # debug end
            remote.send((ob, reward, done, info['obs20'], info['obs2'], info['score2'], info['obs1x'], info['obs2x']))
            # render begin
            reward_sum += reward
            i += 1
            if render and i%10==0:
                env.render('human')
            # render end
        elif cmd == 'reset_ma':
            ob, obs20, obs2, obs2x = env.reset(restart=data)
            remote.send((ob, obs20, obs2, obs2x))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.action_space, env.observation_space))
        else:
            raise NotImplementedError

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, render=False):
        """
        envs: list of gym environments to run in subprocesses
        """
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        
        if not render:
            self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn)) )
                for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]
        else:
            self.ps = [Process(target=worker, 
                               args=(work_remote, CloudpickleWrapper(env_fn), i==0) )
                       for (work_remote, env_fn, i) in zip(self.work_remotes, env_fns, range(nenvs))]
        
        for p in self.ps:
            p.start()

        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self.remotes[0].recv()


    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos
    
    def step_ma(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step_ma', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, obs20, obs2, score2, obs1x, obs2x = zip(*results)
        infos = dict(obs20=np.stack(obs20), obs2=np.stack(obs2), score2=np.stack(score2), 
                        obs1x=np.array((None,*obs1x)), obs2x=np.array((None,*obs2x)),
                        )
        return np.stack(obs), np.stack(rews), np.stack(dones), infos
    
    def reset_ma(self, restart=0):
        for remote in self.remotes:
            remote.send(('reset_ma', restart))
        results = [remote.recv() for remote in self.remotes]
        obs, obs20, obs2, obs2x = zip(*results)
        return np.stack(obs), np.stack(obs20), np.stack(obs2), np.array((None,*obs2x))
    
    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    @property
    def num_envs(self):
        return len(self.remotes)
