from MAAC.MAAC import MAAC
from MAAC.params import scale_reward
import numpy as np
import torch as th
import wandb
# import pressureplate
from gym.envs.registration import register
import gym
import argparse

register(
	id='multigrid-soccer-v0',
	entry_point='gym_multigrid.envs:SoccerGame4HEnv10x15N2',
)

register(
	id='multigrid-collect-v0',
	entry_point='gym_multigrid.envs:CollectGame4HEnv10x10N2',
)

# do not render the scene
e_render = False

food_reward = 10.
poison_reward = -1.
encounter_reward = 0.01
n_coop = 2

env_name = 'multigrid-soccer-v0'

world = gym.make(env_name)

# vis = visdom.Visdom(port=5274)
reward_record = []

np.random.seed(1234)
th.manual_seed(1234)

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--n-agents', default=len(world.agents), type=int)
parser.add_argument('--n-states', default=np.prod(world.observation_space.shape), type=int)
parser.add_argument('--n-actions', default=world.action_space.n, type=int)
parser.add_argument('--capacity', default=1000000, type=int)
parser.add_argument('--batch-size', default=1000, type=int)
parser.add_argument('--n-episode', default=int(3e6), type=int)
parser.add_argument('--max-steps', default=int(1e8), type=int)
parser.add_argument('--episodes-before-train', default=10000, type=int)
# add eps
parser.add_argument('--eps', default=0.2, type=float)
args = parser.parse_args()

world.reset()

n_agents = args.n_agents
n_states = args.n_states
n_actions = args.n_actions
capacity = args.capacity
batch_size = args.batch_size
n_episode = args.n_episode
max_steps = args.max_steps
episodes_before_train = args.episodes_before_train

# n_agents = len(world.agents)
# n_states = np.prod(world.observation_space.shape) #TODO 이것도 수정했습니다.
# n_actions = world.action_space.n
# capacity = 1000000
# batch_size = 1000
# n_episode = int(3e6)
# max_steps = 1000
# episodes_before_train = 100

win = None
param = None

maddpg = MAAC(n_agents, n_states, n_actions, batch_size, capacity,
			  episodes_before_train, epsilon=args.eps)
wandb.init(project="baebae_0409", config=args.__dict__)
wandb.run.name = f"baebaerun_maac"

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
# for i_episode in range(n_episode):
i_episode = 0
while True:
	obs = world.reset()
	obs = np.stack(obs)
	if isinstance(obs, np.ndarray):
		obs = th.from_numpy(obs).float()
	total_reward = 0.0
	rr = np.zeros((n_agents,))
	t = 0
	while True:
		log = {}
		# render every 100 episodes to speed up training
		if i_episode % 20 == 0 and e_render:
			world.render()
		obs = obs.type(FloatTensor)
		# TODO: action space 어떻게 구성된거임
		# action = maddpg.select_action(obs).data.cpu()
		# actions = {agent: maddpg.select_action(obs[agent]).data.cpu().numpy() for agent in
		#            world.agents}  #
		agents_actions = maddpg.select_action(obs.reshape((n_agents, -1))).data.cpu().numpy()
		# agents_actions = maddpg.select_action(obs).data.cpu().numpy()
		actions = np.argmax(agents_actions, axis=1)

		obs_, reward, done, _ = world.step(actions)
		# obs_, reward, done, _ = world.step((action*0.01).numpy())

		reward = np.stack(reward)
		reward = th.from_numpy(reward).float()
		if t != max_steps - 1:
			next_obs = obs_
			next_obs = np.stack(next_obs)
			if isinstance(next_obs, np.ndarray):
				next_obs = th.from_numpy(next_obs).float()
		else:
			next_obs = None

		# subtract reward with t*1e-3
		reward = reward - t * 1e-3

		total_reward += reward.sum()
		rr += reward.cpu().numpy()

		# make actions to be one-hot vectors
		actions = np.eye(n_actions)[actions]

		maddpg.memory.push(obs.data, th.from_numpy(np.stack(actions)).float(), next_obs, reward)
		obs = next_obs

		c_loss, a_loss = maddpg.update_policy()

		# add reward, c_loss, a_loss to log
		log['t/reward'] = reward.sum()
		log['t/c_loss'] = c_loss
		log['t/a_loss'] = a_loss
		if reward.sum() > 0:
			print('reward: ', reward.sum())
		wandb.log(log)

		t += 1

		if done or (t > max_steps - 1):
			obs = world.reset()
			obs = np.stack(obs)
			if isinstance(obs, np.ndarray):
				obs = th.from_numpy(obs).float()
			obs = obs.type(FloatTensor)
			# print('done: {} {} {} {} {}'.format(*done))
			# print('truncated: {} {} {} {} {}'.format(*truncated))
			break

	maddpg.episode_done += 1
	print('Episode: %d, reward = %f' % (i_episode, total_reward))
	reward_record.append(total_reward)
	log['episode/reward'] = reward.sum()
	log['episode/tot_reward'] = total_reward
	wandb.log(log)
	if maddpg.episode_done == maddpg.episodes_before_train:
		print('training now begins...')
		print('MADDPG on WaterWorld\n' +
			  'scale_reward=%f\n' % scale_reward +
			  'agent=%d' % n_agents +
			  ', coop=%d' % n_coop +
			  ' \nlr=0.001, 0.0001, sensor_range=0.3\n')
	i_episode += 1
	if t > max_steps - 1:
		break

wandb.finish()

world.close()