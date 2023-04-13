from MAAC.MAAC import MAAC
from MAAC.params import scale_reward
import numpy as np
import torch as th

from pettingzoo.sisl import pursuit_v4
from magent2.environments import gather_v4, gather_v5


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


# vis = visdom.Visdom(port=5274)
reward_record = []

np.random.seed(1234)
th.manual_seed(1234)
world.reset()
n_agents = len(world.agents)
n_states = np.prod(world.observation_space('omnivore_0').shape) #TODO 이것도 수정했습니다.
n_actions = world.action_space('omnivore_0').n
capacity = 1000000
batch_size = 100

n_episode = 20000
max_steps = 1000
episodes_before_train = 1000

win = None
param = None

magent = MAAC(n_agents, n_states, n_actions, batch_size, capacity,
                episodes_before_train)

FloatTensor = th.cuda.FloatTensor if magent.use_cuda else th.FloatTensor
for i_episode in range(n_episode):
    obs = world.reset()
    # obs = world.last()

    obs = np.stack(obs.values())
    # obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()
    total_reward = 0.0
    rr = np.zeros((n_agents,))
    for t in range(max_steps):
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

        total_reward += reward.sum()
        rr += reward.cpu().numpy()
        magent.memory.push(obs, action, next_obs, reward)
        obs = next_obs

        c_loss, a_loss = magent.update_policy()

        if done:
            # print('done: {} {} {} {} {}'.format(*done))
            # print('truncated: {} {} {} {} {}'.format(*truncated))
            break
    magent.episode_done += 1
    print('Episode: %d, reward = %f' % (i_episode, total_reward))
    reward_record.append(total_reward)

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

world.close()