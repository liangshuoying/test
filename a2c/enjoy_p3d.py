import argparse
import os

import torch
from torch.autograd import Variable

#~ from envs import make_env
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from asteroids import make_env
#~ from astmaze import make_env


parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--algo', default='a2c',
                    help='algorithm to use: a2c | ppo | acktr')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
#~ parser.add_argument('--num-stack', type=int, default=4,
parser.add_argument('--num-stack', type=int, default=1,
                    help='number of frames to stack (default: 4)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='LunarLanderContinuous-v2',
                    help='environment to train on (default: PongNoFrameskip-v4)')
#~ parser.add_argument('--load-dir', default='./trained_models/',
parser.add_argument('--load-dir', default='./trained_models/a2c',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--log-dir', default='./tmp/gym/',
                    help='directory to save agent logs (default: /tmp/gym)')
args = parser.parse_args()

#~ MAX_EPISODE = 100
#~ MAX_EPISODE = 50
#~ MAX_EPISODE = 20
MAX_EPISODE = 1000

#~ args.algo = 'a2c'
args.algo = 'acktr'
#~ args.algo = 'ppo'
#~ args.load_dir = os.path.join('./trained_models/', args.algo)

model_name = "acktr_20180223_103324/EP2870_0h7m_430"

args.env_name = "Panda3d_ast"
# args.env_name = "LunarLanderContinuous-v2"
#~ args.env_name = "BipedalWalker-v2"
#~ args.env_name = "BipedalWalkerHardcore-v2"
args.num_stack = 1

args.load_dir = os.path.join('./trained_models/', args.env_name.replace('-', '_'))


conf = args.__dict__
    
env_args = dict(obs_pixel=0, obs_size=6, act_disc=0, obs_win=4, obs_dtype=None)
#~ conf['env_args'] = dict(obs_pixel=1, act_disc=0, obs_win=4, obs_dtype=None)
#~ conf['env_args'].update(close=1)


env = make_env(args.env_name, args.seed, 0, args.log_dir, env_args)()


actor_critic = torch.load(os.path.join(args.load_dir, model_name + ".pt"))
#~ actor_critic = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))
#~ print(actor_critic)
#~ actor_critic.cpu()
actor_critic.eval()

obs_shape = env.observation_space.shape
# py3 style
#     obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
# py27 style
obs_shape = (obs_shape[0] * args.num_stack,) + tuple(obs_shape[1:])
current_state = torch.zeros(1, *obs_shape)
#~ current_state = current_state.cuda()


def update_current_state(state):
    shape_dim0 = env.observation_space.shape[0]
    state = torch.from_numpy(state).float()
    if args.num_stack > 1:
        current_state[:, :-shape_dim0] = current_state[:, shape_dim0:]
    current_state[:, -shape_dim0:] = state

#~ env.render('human')
state = env.reset()
update_current_state(state)

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

steps = 0
episode = 0
episode_rewards = 0
sum_rewards = 0
sum_steps = 0
reward_min = None
reward_max = None

while True:
    steps += 1
    
    value, action = actor_critic.act(Variable(current_state, volatile=True),
                                        deterministic=True)
    #~ cpu_actions = action.data.squeeze(1).cpu().numpy()
    cpu_actions = action.data.cpu().numpy()
     # Clip inplace for LunarLanderContinuous
    cpu_actions.clip(-1, 1, cpu_actions)
    # Clip End
    # Obser reward and next state
    state, reward, done, _ = env.step(cpu_actions[0])
    
    episode_rewards += reward
    
    #~ if steps%100==0:
        #~ print("reward", reward)
        #~ print("state", state)
    
    if args.env_name.find('Bullet') > -1:
        if torsoId > -1:
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)
    
    if steps%2==0:
        env.render('human')

    if done:
        print('-episode_rewards %.0f steps %d'%(episode_rewards, steps))
        sum_rewards += episode_rewards
        sum_steps += steps
        reward_min = episode_rewards if reward_min is None else min(reward_min, episode_rewards)
        reward_max = episode_rewards if reward_max is None else max(reward_max, episode_rewards)
        steps = 0
        episode_rewards = 0
        episode += 1
        if episode>MAX_EPISODE:
            break
        state = env.reset()
        actor_critic = torch.load(os.path.join(args.load_dir, model_name + ".pt"))
        #~ actor_critic = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))
        actor_critic.eval()

    update_current_state(state)

print("average_rewards:{:.0f} avg_step[{:.0f}]".format(sum_rewards/episode, sum_steps/episode))
print("episode_rewards: min:{:.0f} max:{:.0f}".format(reward_min, reward_max))


