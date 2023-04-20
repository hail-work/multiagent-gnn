from .model import Critic, ActorMAAC, PerceptGNN, SGActor, SGCritic, SAGEGActor, SAGEGCritic
import torch as th
from copy import deepcopy
from .memory import ReplayMemory, Experience
from torch.optim import Adam
from .randomProcess import OrnsteinUhlenbeckProcess
import torch.nn as nn
import numpy as np
from .params import scale_reward
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.utils import to_networkx
import networkx as nx

def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class MAFO_GAC: # Multi-Agent fully observable actorcritic
    def __init__(self, n_agents, observation_shape, dim_act, batch_size,
                 capacity, episodes_before_train, epsilon=0.1):
        dim_obs = np.prod(observation_shape)
        # self.actors_gnn = [PerceptGNN(observation_shape) for i in range(n_agents)]
        self.actors = [SAGEGActor(observation_shape, dim_act) for i in range(n_agents)]

        # self.critics_gnn = PerceptGNN((*observation_shape[:2],n_agents*observation_shape[2]))
        # self.critics_gnn = PerceptGNN((*observation_shape[:2],n_agents*observation_shape[2]))
        self.critics = SAGEGCritic(n_agents, (*observation_shape[:2],n_agents*observation_shape[2]),
                               dim_act)
        # self.critics = Critic(n_agents, dim_obs,
        #                        dim_act)
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)
        # self.critics_gnn_target = deepcopy(self.critics_gnn)
        # self.actors_gnn_target = deepcopy(self.actors_gnn)
        # self.actors_target_ = deepcopy(self.actors_)
        # self.critics_target_ = deepcopy(self.critics_)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.use_cuda = th.cuda.is_available()
        self.episodes_before_train = episodes_before_train

        self.GAMMA = 0.95
        self.tau = 0.01
        self.epsilon = epsilon

        self.var = [1.0 for i in range(n_agents)]
        self.critic_optimizer = Adam(self.critics.parameters(),
                                      lr=0.01)
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.001) for x in self.actors]

        if self.use_cuda:
            # for x in self.actors_gnn:
            #     x.cuda()
            # for x in self.actors_gnn_target:
            #     x.cuda()
            # self.critics_gnn.cuda()
            # self.critics_gnn_target.cuda()

            for x in self.actors:
                x.cuda()
            self.critics.cuda()
            # for x in self.critics:
                # x.cuda()
            for x in self.actors_target:
                x.cuda()
            self.critics_target.cuda()
            # for x in self.critics_target:
            #     x.cuda()

        self.steps_done = 0
        self.episode_done = 0

    def update_policy(self):
        # do not train until exploration is enough
        if self.episode_done <= self.episodes_before_train:
            return None, None

        ByteTensor = th.cuda.ByteTensor if self.use_cuda else th.ByteTensor
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor

        c_loss = []     # Critic loss
        a_loss = []     # Actor loss
        for agent in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)       # transition Data
            batch = Experience(*zip(*transitions))
            non_final_mask = ByteTensor(list(map(lambda s: s is not None,
                                                 batch.next_states)))
            # state_batch: batch_size x n_agents x dim_obs            # transition Data 에서 batch_size 만큼 추출
            state_batch = th.stack(batch.states).type(FloatTensor)    # state
            action_batch = th.stack(batch.actions).type(FloatTensor)  # action
            reward_batch = th.stack(batch.rewards).type(FloatTensor)  # reward
            # : (batch_size_non_final) x n_agents x dim_obs
            non_final_next_states = th.stack(
                [s for s in batch.next_states
                 if s is not None]).type(FloatTensor)

            state_shape = state_batch.shape
            # for current agent
            whole_state_ = state_batch.unsqueeze(-1).swapaxes(1,-1).squeeze(1).reshape(state_shape[0],state_shape[2]*state_shape[3],state_shape[1]*state_shape[4])

            # set x_ that to be append
            x_cc, adj_cc, mask_cc = getg_xadjmask_batch(whole_state_, state_shape[2], state_shape[3])

            if self.use_cuda:
                x_cc = x_cc.cuda()
                adj_cc = adj_cc.cuda()
                mask_cc = mask_cc.cuda()

            # whole_state = self.critics_gnn(x_,adj_,mask_)
            # whole_state = state_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)
            self.critic_optimizer.zero_grad()
            # current_Q = self.critics(whole_state, whole_action)      # Critic에서 온 Q-value
            # current_Q = self.critics(whole_state.view(self.batch_size,-1), whole_action)
            current_Q = self.critics(whole_action, x_cc,adj_cc,mask_cc)

            non_final_next_states_ = non_final_next_states[:, 0, :].reshape(self.batch_size, -1, state_shape[-1])

            non_final_next_actions = []
            for i in range(self.n_agents):
                # set x_ that to be append
                x_, adj_, mask_ = getg_xadjmask_batch(non_final_next_states_, state_shape[2], state_shape[3])

                if self.use_cuda:
                    x_ = x_.cuda()
                    adj_ = adj_.cuda()
                    mask_ = mask_.cuda()

                # next_state = self.actors_gnn_target[i](x_, adj_, mask_)
                # non_final_next_actions.append(
                #     self.actors_target[i](next_state)
                # )
                non_final_next_actions.append(
                    self.actors_target[i](x_, adj_, mask_)
                )
            # non_final_next_actions = [
            #     self.actors_target_[i](non_final_next_states[:,
            #                                                 i,
            #                                                 :]) for i in range(
            #                                                     self.n_agents)]
            non_final_next_actions = th.stack(non_final_next_actions)
            non_final_next_actions = (
                non_final_next_actions.transpose(0,
                                                 1).contiguous())

            target_Q = th.zeros(
                self.batch_size).type(FloatTensor)                         # Target- Network 에서 온 Q-value

            # target_Q[non_final_mask] = self.critics_target[agent](
            #     non_final_next_states.view(-1, self.n_agents * self.n_states),
            #     non_final_next_actions.view(-1,
            #                                 self.n_agents * self.n_actions)
            # ).squeeze()
            next_state_ = non_final_next_states.unsqueeze(-1).swapaxes(1,-1).squeeze(1).reshape(state_shape[0],state_shape[2]*state_shape[3],state_shape[1]*state_shape[4])

            # set x_ that to be append
            x_, adj_, mask_ = getg_xadjmask_batch(next_state_, state_shape[2], state_shape[3])

            if self.use_cuda:
                x_ = x_.cuda()
                adj_ = adj_.cuda()
                mask_ = mask_.cuda()

            # whole_next_state_ = self.critics_gnn_target(x_,adj_,mask_)
            # target_Q = self.critics_(whole_next_state_.view(self.batch_size,-1), whole_action.view(self.batch_size, -1)).squeeze()
            # target_Q[non_final_mask] = self.critics_target(whole_next_state_.view(self.batch_size,-1), non_final_next_actions.view(self.batch_size, -1)).squeeze()
            target_Q[non_final_mask] = self.critics_target(non_final_next_actions.view(self.batch_size, -1), x_,adj_,mask_).squeeze()


            # target_Q[non_final_mask] = self.critics_target_(
            #     non_final_next_states.view(-1, self.n_agents * self.n_states),
            #     non_final_next_actions.view(-1,
            #                                 self.n_agents * self.n_actions)
            # ).squeeze()
            # scale_reward: to scale reward in Q functions

            target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (
                reward_batch[:, agent].unsqueeze(1) * scale_reward)

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())            # Critic-Loss
            loss_Q.backward()
            # self.critic_optimizer.step()
            self.critic_optimizer.step()

            # self.actor_optimizer[agent].zero_grad()
            self.actor_optimizer[agent].zero_grad()                  # Actor 학습
            state_i = state_batch[:, agent, :]
            # action_i = self.actors[agent](state_i)
            state_i = state_i.reshape(self.batch_size, -1, state_shape[-1])

            action_i = []

            for i in range(self.batch_size):
                x, adj, mask = getg_xadjmask(state_i[i,:,:], state_shape[2], state_shape[3])

                if self.use_cuda:
                    x = x.cuda()
                    adj = adj.cuda()
                    mask = mask.cuda()

                # state_ = self.actors_gnn[agent](x, adj, mask)
                action_i.append(self.actors[agent](x, adj, mask).squeeze(0))

            action_i = th.stack(action_i)
            # action_i = self.actors_[agent](state_i)
            ac = action_batch.clone()
            ac[:, agent, :] = action_i
            whole_action = ac.view(self.batch_size, -1)
            # actor_loss = -self.critics(whole_state, whole_action)
            actor_loss = -self.critics(whole_action, x_cc,adj_cc,mask_cc)
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            # self.actor_optimizer[agent].step()
            self.actor_optimizer[agent].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        if self.steps_done % 100 == 0 and self.steps_done > 0:
            soft_update(self.critics_target, self.critics, self.tau)
            for i in range(self.n_agents):
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss

    def select_action(self, state_batch):
        # state_batch: n_agents x state_dim
        actions = th.zeros(
            self.n_agents,
            self.n_actions)
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        for i in range(self.n_agents):


            sb = state_batch[i, :].detach()
            x, adj, mask = getg_xadjmask(sb)

            if self.use_cuda:
                x = x.cuda()
                adj = adj.cuda()
                mask = mask.cuda()

            act = self.actors[i](x,adj,mask).squeeze()

            act += th.from_numpy(
                np.random.rand(self.n_actions) * self.var[i]).type(FloatTensor)

            # summed to one
            act /= act.sum()

            if self.episode_done > self.episodes_before_train and\
               self.var[i] > 0.05:
                self.var[i] *= 0.999998
            act = th.clamp(act, 0, 1.0)

            # random action part
            act_i = np.random.choice(np.linspace(0, self.n_actions - 1, self.n_actions).astype(np.int16),
                             p=act.to('cpu').detach().numpy())

            # if it requires one hot vector encoding
            _act = np.zeros((self.n_actions))
            _act[act_i] = 1
            act = th.from_numpy(_act)
            actions[i, :] = act

            # if random number is smaller than epsilon, do random action
            if np.random.rand() < self.epsilon:
                act = th.from_numpy(np.random.rand(self.n_actions))
                act /= act.sum()
            actions[i, :] = act
        self.steps_done += 1

        return actions

def getg_xadjmask_batch(sbs, x_size=10,y_size=10):
    x_ = th.Tensor(0,x_size*y_size,sbs.shape[-1])
    adj_ = th.Tensor(0,x_size*y_size,x_size*y_size)
    mask_ = th.Tensor(0,x_size*y_size)

    for sb in sbs:
        x_size = np.sqrt(sb.shape[0]).astype(np.int16)
        y_size = np.sqrt(sb.shape[0]).astype(np.int16)

        G = nx.grid_2d_graph(x_size, y_size)
        G.add_edges_from([edge for edge in G.edges], weight=1)
        G.add_edges_from([
                             ((x, y), (x + 1, y + 1))
                             for x in range(x_size - 1)
                             for y in range(y_size - 1)
                         ] + [
                             ((x + 1, y), (x, y + 1))
                             for x in range(x_size - 1)
                             for y in range(y_size - 1)
                         ], weight=1.4)
        for node in G.nodes:
            G.nodes[node]['x'] = sb.reshape(x_size, y_size, -1)[node[0], node[1]].detach().cpu().numpy()

        pyg = from_networkx(G)

        x = pyg.x.view(1, x_size*y_size, -1)
        adj = to_dense_adj(pyg.edge_index, max_num_nodes=pyg.num_nodes)
        # make mask that is filled with True
        mask = np.ones((x_size, y_size))
        mask = th.Tensor(mask.reshape(1, -1))

        x_ = th.cat((x_, x), dim=0)
        adj_ = th.cat((adj_, adj), dim=0)
        mask_ = th.cat((mask_, mask), dim=0)
    return x_, adj_, mask_

def getg_xadjmask(sb, x_size=10, y_size=10):
    # x_size = np.sqrt(sb.shape[0]/2).astype(np.int16)
    # y_size = np.sqrt(sb.shape[0]/2).astype(np.int16)

    import networkx as nx

    G = nx.grid_2d_graph(x_size, y_size)

    for edge in G.edges:
        G.edges[edge]['weight'] = 1

    G.add_edges_from([
                         ((x, y), (x + 1, y + 1))
                         for x in range(x_size - 1)
                         for y in range(y_size - 1)
                     ] + [
                         ((x + 1, y), (x, y + 1))
                         for x in range(x_size - 1)
                         for y in range(y_size - 1)
                     ], weight=1)

    for x in range(x_size):
            for y in range(y_size):
                # print x, y
                G.nodes[(x,y)]['x'] = sb.reshape(x_size, y_size,-1)[x, y].detach().to('cpu').numpy()

    pyg = from_networkx(G)

    x = pyg.x.view(1, x_size*y_size, -1)
    adj =  to_dense_adj(pyg.edge_index, max_num_nodes=pyg.num_nodes)
    # make mask that is filled with True
    mask = np.ones((x_size, y_size))
    mask = th.Tensor(mask.reshape(1, -1))

    return x, adj, mask

if __name__=="__main__":
    x_ = th.Tensor(0,100,6)
    adj_ = th.Tensor(0,100,100)
    mask_ = th.Tensor(0,100)

    sbs = th.Tensor(np.zeros((2,100,6)))

    for sb in sbs:
        x_size = np.sqrt(sb.shape[0]).astype(np.int16)
        y_size = np.sqrt(sb.shape[0]).astype(np.int16)

        import networkx as nx
        G = nx.grid_2d_graph(x_size, y_size)
        G.add_edges_from([edge for edge in G.edges], weight=1)
        G.add_edges_from([
                             ((x, y), (x + 1, y + 1))
                             for x in range(x_size - 1)
                             for y in range(y_size - 1)
                         ] + [
                             ((x + 1, y), (x, y + 1))
                             for x in range(x_size - 1)
                             for y in range(y_size - 1)
                         ], weight=1.4)
        for node in G.nodes:
            G.nodes[node]['x'] = sb.reshape(x_size, y_size, -1)[node[0], node[1]].detach().numpy()

        pyg = from_networkx(G)

        x = pyg.x.view(1, x_size*y_size, -1)
        adj = to_dense_adj(pyg.edge_index, max_num_nodes=pyg.num_nodes)
        # make mask that is filled with True
        mask = np.ones((x_size, y_size))
        mask = th.Tensor(mask.reshape(1, -1))

        x_ = th.cat((x_, x), dim=0)
        adj_ = th.cat((adj_, adj), dim=0)
        mask_ = th.cat((mask_, mask), dim=0)


if __name__=="__main2__":
    sb = th.Tensor(np.zeros(200))
    x_size = np.sqrt(sb.shape[0]/2).astype(np.int16)
    y_size = np.sqrt(sb.shape[0]/2).astype(np.int16)

    import networkx as nx

    G = nx.grid_2d_graph(x_size, y_size)

    for edge in G.edges:
        G.edges[edge]['weight'] = 1

    G.add_edges_from([
                         ((x, y), (x + 1, y + 1))
                         for x in range(x_size - 1)
                         for y in range(y_size - 1)
                     ] + [
                         ((x + 1, y), (x, y + 1))
                         for x in range(x_size - 1)
                         for y in range(y_size - 1)
                     ], weight=1.4)

    for x in range(10):
            for y in range(10):
                # print x, y
                G.nodes[(x,y)]['x'] = sb.reshape(x_size, y_size,-1)[x, y].detach().numpy()

    pyg = from_networkx(G)

    x = pyg.x.view(1, x_size*y_size, -1)
    adj =  to_dense_adj(pyg.edge_index, max_num_nodes=pyg.num_nodes)
    # make mask that is filled with True
    mask = np.ones((x_size, y_size))
    mask = th.Tensor(mask.reshape(1, -1))