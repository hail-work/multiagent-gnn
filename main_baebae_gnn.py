from MAAC.MAGAC import MAFO_GAC
from MAAC.params import scale_reward
import numpy as np
import torch as th
import wandb
# import pressureplate
from gym.envs.registration import register
import gym
import argparse
import cv2
from tqdm import tqdm


def obs_to_fobs(obs):
	obs = np.stack(obs)
	_fobs = np.zeros((*obs.shape[0:3], 2))

	for j in range(obs.shape[0]):
		_fobs[j, :, :, 0] = obs[0, :, :, 0]
		for i in range(obs.shape[0]):
			_fobs[j, :, :, 1] += obs[0, :, :, 5] * (i+1)

	return _fobs

register(
	id='multigrid-soccer-v0',
	entry_point='gym_multigrid.envs:SoccerGame4HEnv10x15N2',
)

register(
	id='multigrid-collect-v0',
	entry_point='gym_multigrid.envs:CollectGame4HEnv10x10N2',
)

# do not render the scene
e_render = True

food_reward = 10.
poison_reward = -1.
encounter_reward = 0.01
n_coop = 2

env_name = 'multigrid-collect-v0'

world = gym.make(env_name)

# vis = visdom.Visdom(port=5274)
reward_record = []

np.random.seed(1234)
th.manual_seed(1234)

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--n-agents', default=len(world.agents), type=int)
# parser.add_argument('--n-states', default=np.prod(world.observation_space.shape), type=int)
parser.add_argument('--n-states', default=np.prod([*world.observation_space.shape[:2],2]), type=int)
parser.add_argument('--n-actions', default=7, type=int)
parser.add_argument('--capacity', default=100000, type=int)
# parser.add_argument('--batch-size', default=100, type=int)
parser.add_argument('--batch-size', default=10, type=int)
parser.add_argument('--n-episode', default=int(3e6), type=int)
parser.add_argument('--max-steps', default=int(1e8), type=int)
parser.add_argument('--episodes-before-train', default=100, type=int)
# parser.add_argument('--episodes-before-train', default=2, type=int)
# add eps
parser.add_argument('--eps', default=0.1, type=float)
parser.add_argument('--save-vid-every', default=int(1e5), type=int)
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


win = None
param = None

maddpg = MAFO_GAC(n_agents, (*world.observation_space.shape[:-1],2), n_actions, batch_size, capacity,
			  episodes_before_train, epsilon=args.eps)
wandb.init(project="baebae_magac", config=args.__dict__)
wandb.run.name = f"baebaerun_mafo_small"

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
# for i_episode in range(n_episode):
i_episode = 0
for i_episode in tqdm(range(n_episode)):
	obs = world.reset()
	_fobs = obs_to_fobs(obs)
	if isinstance(_fobs, np.ndarray):
		_fobs = th.from_numpy(_fobs).float()

	total_reward_idx = np.zeros(3)
	total_reward = 0.0

	total_c_loss = np.nan
	total_a_loss = np.nan
	rr = np.zeros((n_agents,))
	t = 0

	# open a file that named "video_{i_episode}.mp4" and to save video
	if (i_episode+1) % args.save_vid_every == 0 and e_render:
		# Create a VideoWriter object
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec
		out = cv2.VideoWriter(f'output_{i_episode}.mp4', fourcc, 20.0, (640, 480))  # output file name, codec, fps, resolution


	while True:
		log = {}

		# save video for every 100 episodes
		if (i_episode+1) % args.save_vid_every == 0 and e_render:
			frame = world.render()
			# write a video frame to out file
			out.write(frame)
		# check all three of obs[:,0,0,0] are same
		# np.all(obs[:, 0, 0, 0] == obs[0, 0, 0, 0])



		obs = _fobs.type(FloatTensor)
		agents_actions = maddpg.select_action(obs.reshape((n_agents, -1))).data.cpu().numpy()
		# agents_actions = maddpg.select_action(obs).data.cpu().numpy()
		actions = np.argmax(agents_actions, axis=1)

		obs_, reward, done, _ = world.step(actions)
		# obs_, reward, done, _ = world.step((action*0.01).numpy())
		total_reward_idx+=reward
		if reward.sum() != 0:
			# fancy print
			print("reward: ", reward)
			if reward.sum()>1:
				print("reward sum: ", reward.sum())
				print("wow")
		# argmax of actions
		done_actions = np.all(actions==6)
		if done_actions:
			done = True
			# if they decided to finish the episode before the best possible reward
			if total_reward_idx.sum() != len(world.agents)*2:
				reward -= 2 # this is for not to quit before it collect all
			else:
				pass
		if total_reward_idx.sum() == len(world.agents)*2: # if it collected all
			done = True


		reward = np.stack(reward)
		reward = th.from_numpy(reward).float()
		if t != max_steps - 1:
			next_obs = obs_
			next_obs = obs_to_fobs(next_obs)
			if isinstance(next_obs, np.ndarray):
				next_obs = th.from_numpy(next_obs).float()
		else:
			next_obs = None

		# subtract reward with t*1e-3
		reward = reward - t * 1e-7

		total_reward += reward.sum()
		rr += reward.cpu().numpy()

		# make actions to be one-hot vectors
		actions = np.eye(n_actions)[actions]

		maddpg.memory.push(obs.data.to('cpu'), th.from_numpy(np.stack(actions)).float(), next_obs, reward)
		obs = next_obs

		c_loss, a_loss = maddpg.update_policy()
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
		if reward.sum() > 0:
			print('reward: ', reward.sum())

		# add reward, c_loss, a_loss to log
		log['t/reward'] = reward.sum()
		# if reward.sum() > 0:
		# 	print('reward: ', reward.sum())
		wandb.log(log)

		t += 1

		if t>int(1e4):
			print('truncated')

		if done or (t > max_steps - 1):
			obs = world.reset()
			obs = np.stack(obs)
			if isinstance(obs, np.ndarray):
				obs = th.from_numpy(obs).float()
			obs = obs.type(FloatTensor)
			# print('done: {} {} {} {} {}'.format(*done))
			# print('truncated: {} {} {} {} {}'.format(*truncated))
			break

	# open a file that named "video_{i_episode}.mp4" and to save video
	if (i_episode+1) % args.save_vid_every == 0 and e_render:
		# Release the resources
		out.release()
		cv2.destroyAllWindows()
		# empty Capture object

	maddpg.episode_done += 1
	print('Episode: %d, reward = %f' % (i_episode, total_reward))
	reward_record.append(total_reward)
	log['episode/reward'] = reward.sum()
	log['episode/tot_reward'] = total_reward
	if not np.isnan(total_c_loss):
		log['episode/c_loss']=total_c_loss
		log['episode/a_loss']=total_a_loss
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