from MAAC.MAAC import MAAC
from MAAC.params import scale_reward
import numpy as np
import torch as th

from pettingzoo.sisl import pursuit_v4
from magent2.environments import gather_v4, gather_v5

import wandb

from pettingzoo.utils.conversions import to_parallel

# do not render the scene
e_render = True

food_reward = 10.
poison_reward = -1.
encounter_reward = 0.01
n_coop = 2


np.random.seed(1234)
th.manual_seed(1234)
# world = pursuit_v4.env(max_cycles=500, x_size=16, y_size=16, shared_reward=True, n_evaders=30,
# n_pursuers=8,obs_range=7, n_catch=2, freeze_evaders=False, tag_reward=0.01,
# catch_reward=5.0, urgency_reward=-0.1, surround=True, constraint_window=1.0)

world = gather_v5.env(minimap_mode=False, step_reward=-0.01, attack_penalty=-0.1,
dead_penalty=-1, attack_food_reward=0.5, max_cycles=500, extra_features=False)
world.reset(seed = 1)
# world.render()


world = to_parallel(world)

# parser for every importat parameters
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n_episode', type=int, default=20000)
parser.add_argument('--max_steps', type=int, default=int(1e8))
parser.add_argument('--episodes_before_train', type=int, default=1000)
parser.add_argument('--capacity', type=int, default=1000000)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--wandb_name', type=str, default='maac')
parser.add_argument('--epsilon', type=float, default=0.2)

args = parser.parse_args()

capacity = args.capacity
batch_size = args.batch_size
n_episode = args.n_episode
max_steps = args.max_steps
episodes_before_train = args.episodes_before_train
epsilon = args.epsilon



# vis = visdom.Visdom(port=5274)
reward_record = []

np.random.seed(1234)
th.manual_seed(1234)
world.reset()
n_agents = len(world.agents)
n_states = np.prod(world.observation_space('omnivore_0').shape) #TODO 이것도 수정했습니다.
n_actions = world.action_space('omnivore_0').n
t = 0

win = None
param = None

magent = MAAC(n_agents, n_states, n_actions, batch_size, capacity,
                episodes_before_train)
wandb.init(project="jin_0413",config=args.__dict__)
wandb.run.name = args.wandb_name

FloatTensor = th.cuda.FloatTensor if magent.use_cuda else th.FloatTensor
i_episode=0
while True:
    obs = world.reset()
    # obs = world.last()

    obs = np.stack(obs.values())
    # obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()
    total_reward = 0.0
    total_c_loss = np.nan
    total_a_loss = np.nan
    rr = np.zeros((n_agents,))
    while True:
        log = {}
        # render every 100 episodes to speed up training
        if i_episode % 20 == 0 and e_render:
            world.render()
        obs = obs.type(FloatTensor)


        action = magent.select_action(obs).data.cpu()
        actions = {agent: np.argmax(action[count].numpy()) for count, agent in enumerate(world.agents)}


        # actions = {agent: magent.select_action(obs[count]).data.cpu().numpy() for count, agent in
        #            enumerate(world.agents)}  #
        # agents_actions = magent.select_action(obs.reshape((n_agents,-1))).data.cpu().numpy()
        # # agents_actions = maddpg.select_action(obs).data.cpu().numpy()
        # actions = np.argmax(agents_actions, axis=1)

        obs_, reward, done, truncated, _ = world.step(actions)
        # obs_, reward, done, _ = world.step((action*0.01).numpy())

        reward = np.stack(reward.values())
        # obs = np.stack(obs)
        if isinstance(reward, np.ndarray):
            reward = th.from_numpy(reward).float()

        obs_ = np.stack(obs_.values())
        # obs = np.stack(obs)
        if isinstance(obs_, np.ndarray):
            obs_ = th.from_numpy(obs_).float()


        if t != max_steps - 1:
            next_obs = obs_
        else:
            next_obs = None

        # find that is not 0 in action
        valid_action = (action != 0).sum()

        # 정확히는 가만히 있는게 아닌 경우에만 reward를 디스카운트해야함
        reward = reward - valid_action*1e-3

        total_reward += reward.sum().detach().numpy()

        magent.memory.push(obs, action, next_obs, reward)
        obs = next_obs

        c_loss, a_loss = magent.update_policy()
        if c_loss is not None:
            if np.isnan(total_c_loss):
                total_c_loss = 0
                total_a_loss = 0
                total_c_loss += sum(c_loss).item()
                total_a_loss += sum(a_loss).item()
            else:
                total_c_loss += sum(c_loss).item()
                total_a_loss += sum(a_loss).item()

            log['t/c_loss'] = sum(c_loss).item()
            log['t/a_loss'] = sum(a_loss).item()


        # add reward, c_loss, a_loss to log
        log['t/reward'] = reward.sum()
        if reward.sum() >0:
            print('reward: ', reward.sum())
        wandb.log(log)

        t += 1
        donedone = np.stack(done.values())
        trunctrunc = np.stack(truncated.values())
        if np.any(donedone) or np.all(trunctrunc) or (t>max_steps-1):
            # print('done: {} {} {} {} {}'.format(*done))
            # print('truncated: {} {} {} {} {}'.format(*truncated))
            obs = world.reset()
            log = {}
            obs = np.stack(obs.values())
            # obs = np.stack(obs)
            if isinstance(obs, np.ndarray):
                obs = th.from_numpy(obs).float()
            obs = obs.type(FloatTensor)

            break
    magent.episode_done += 1
    print('Episode: %d, reward = %f' % (i_episode, total_reward))
    # print total t
    print('steps: %d' % t)
    try:
        print('reward: {}, c_loss: {}, a_loss: {}'.format(log['t/reward'].sum(), log['t/c_loss'].sum(), log['t/a_loss'].sum()))
    except:
        # error
        pass

    reward_record.append(total_reward)
    log['episode/reward']=reward.sum()
    log['episode/tot_reward']=total_reward
    log['episode/c_loss']=total_c_loss
    log['episode/a_loss']=total_a_loss
    wandb.log(log)
    if magent.episode_done == magent.episodes_before_train:
        print('training now begins...')
        print('MAGENTS:\n' +
              'scale_reward=%f\n' % scale_reward +
              'agent=%d' % n_agents +
              ', coop=%d' % n_coop +
              ' \nlr=0.001, 0.0001, sensor_range=0.3\n' +
              'food=%f, poison=%f, encounter=%f' % (
                  food_reward,
                  poison_reward,
                  encounter_reward))
    i_episode += 1
    if t>max_steps-1:
        break

world.close()