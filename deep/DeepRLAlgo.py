import os
import ptan
import time
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from MyEnv import MyEnv
from Config import *


class RLNet(nn.Module):
    def __init__(self, obs_size, n_actions=-1, action_size=-1, **kwargs):
        super(RLNet, self).__init__()
        self.action_size = action_size
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.network_type = None  # Q: Q-network, P: Policy-network DP: DDPG

    # This is only for DQN-like network, i.e. the output of the network is Q value
    @torch.no_grad()
    def evaluate_expected_reward(self, env: MyEnv, evaluate_episodes_for_eval=10, gamma=GAMMA, replay_size=REPLAY_SIZE,
                                 threshold=2.0):
        if self.network_type == 'Q':
            # For Q-network
            selector = ptan.actions.ArgmaxActionSelector()
            agent = ptan.agent.DQNAgent(self, selector, device=device)
            exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=gamma)
            buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=replay_size)

            step = 0
            episode = 0
            evaluate_reward = 0  # Used for evaluate model's performance

            while True:
                step += 1
                buffer.populate(1)

                for reward, steps in exp_source.pop_rewards_steps():
                    episode += 1
                    evaluate_reward += reward
                if episode >= evaluate_episodes_for_eval:
                    break
            evaluate_reward /= evaluate_episodes_for_eval
            return evaluate_reward
        elif self.network_type == 'DP':
            # We do not implement this method for policy network
            evaluate_rewards = 0.0
            for _ in range(evaluate_episodes_for_eval):
                obs = env.reset()
                steps = 0
                while True:
                    obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
                    mu_v = self(obs_v)
                    action = mu_v.squeeze(dim=0).data.cpu().numpy()
                    action = np.clip(action, -threshold, threshold)
                    obs, reward, done, _ = env.step(action)
                    evaluate_rewards += reward
                    steps += 1
                    if steps >= 200:
                        break
                    if done:
                        break
            evaluate_rewards /= evaluate_episodes_for_eval
            return evaluate_rewards
        return


class MLP_Q_Net(RLNet):
    def __init__(self, obs_size, n_actions, hidden_size):
        super(MLP_Q_Net, self).__init__(obs_size, n_actions)
        self.network_type = 'Q'
        self.net = nn.Sequential(
            nn.Linear(self.obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.n_actions)
        )

    def forward(self, x):
        return self.net(x.float())


@torch.no_grad()
def unpack_batch(batch, net, gamma):
    states = []
    actions = []
    rewards = []
    done_masks = []
    last_states = []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        done_masks.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    last_states_v = torch.tensor(last_states).to(device)
    last_state_q_v = net(last_states_v)
    best_last_q_v = torch.max(last_state_q_v, dim=1)[0]
    best_last_q_v[done_masks] = 0.0
    return states_v, actions_v, best_last_q_v * gamma + rewards_v


def unpack_batch_ddqn(batch, device="cpu"):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    states_v = ptan.agent.float32_preprocessor(states).to(device)
    actions_v = ptan.agent.float32_preprocessor(actions).to(device)
    rewards_v = ptan.agent.float32_preprocessor(rewards).to(device)
    last_states_v = ptan.agent.float32_preprocessor(last_states).to(device)
    dones_t = torch.BoolTensor(dones).to(device)
    return states_v, actions_v, rewards_v, dones_t, last_states_v


# Use epsilon-greedy to explore
def double_DQN(env: gym.Env,
               net_extra_hyperparameters: dict,
               net_class=MLP_Q_Net,
               epsilon=INIT_EPSILON,
               epsilon_decay=EPS_DECAY,
               sync_interval=TGT_NET_SYNC,
               gamma=GAMMA,
               batch_size=BATCH_SIZE,
               lr=LR,
               evaluate_episodes_for_conv=EVALUATE_EPISODES_FOR_CONV,
               solve_criterion=SOLVE_CRITERION,
               replay_size=REPLAY_SIZE,
               continue_train_flag=False,
               continue_train_para=None,
               train_episodes_upper_bound=TRAIN_EPISODES_UB,
               silent_flag=True):

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    if continue_train_flag:
        net = net_class(obs_size=obs_size, n_actions=n_actions, **net_extra_hyperparameters)
        net.load_state_dict(continue_train_para)
        net = net.to(device)
    else:
        net = net_class(obs_size=obs_size, n_actions=n_actions, **net_extra_hyperparameters).to(device)
    tgt_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.ArgmaxActionSelector()
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=epsilon, selector=selector)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=gamma)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=replay_size)
    optimizer = optim.Adam(net.parameters(), lr)

    step = 0
    episode = 0
    evaluate_reward = 0  # Used for evaluate model's performance
    solved = False

    while True:
        step += 1
        buffer.populate(1)

        for reward, steps in exp_source.pop_rewards_steps():
            episode += 1
            evaluate_reward += reward
            if not silent_flag:
                print("%d: episode %d done, reward=%.3f, epsilon=%.2f" % (
                    step, episode, reward, selector.epsilon))
            if episode % evaluate_episodes_for_conv == 0:
                evaluate_reward /= evaluate_episodes_for_conv
                solved = evaluate_reward > solve_criterion
                evaluate_reward = 0
        if solved:
            if not silent_flag:
                print("Congrats!")
            break
        if episode >= train_episodes_upper_bound:
            if not silent_flag:
                print("Exceed episode upper bound!")
            break

        if len(buffer) < 2 * batch_size:
            continue

        batch = buffer.sample(batch_size)
        states_v, actions_v, tgt_q_v = unpack_batch(batch, tgt_net.target_model, gamma)
        optimizer.zero_grad()
        q_v = net(states_v)
        q_v = q_v.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        loss_v = F.mse_loss(q_v, tgt_q_v)
        loss_v.backward()
        optimizer.step()
        selector.epsilon *= epsilon_decay

        if step % sync_interval == 0:
            tgt_net.sync()
    # You need to reconstruct buffer using buffer.buffer
    return net, buffer.buffer


# This function is disapproved, we leave it here for history code
# Evaluate a DQN's reward
def evaluate_DQN(env, net, evaluate_episodes_for_eval=10, gamma=GAMMA, replay_size=REPLAY_SIZE):
    selector = ptan.actions.ArgmaxActionSelector()
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=gamma)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=replay_size)

    step = 0
    episode = 0
    evaluate_reward = 0  # Used for evaluate model's performance

    while True:
        step += 1
        buffer.populate(1)

        for reward, steps in exp_source.pop_rewards_steps():
            episode += 1
            evaluate_reward += reward
        if episode >= evaluate_episodes_for_eval:
            break
    evaluate_reward /= evaluate_episodes_for_eval
    return evaluate_reward


class DDPGActor(RLNet):
    def __init__(self, obs_size, act_size, threshold=2.0):
        super(DDPGActor, self).__init__(obs_size=obs_size, action_size=act_size)
        self.network_type = 'DP'
        self.rescale = threshold

        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.rescale * self.net(x)


class DDPGCritic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGCritic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))


class AgentDDPG(ptan.agent.BaseAgent):
    """
    Agent implementing Orstein-Uhlenbeck exploration process
    """
    def __init__(self, net, device="cpu", ou_enabled=True,
                 ou_mu=0.0, ou_teta=0.15, ou_sigma=0.2,
                 ou_epsilon=1.0, threshold=2.0):
        self.net = net
        self.device = device
        self.threshold = threshold
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_teta = ou_teta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon

    def initial_state(self):
        return None

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()

        if self.ou_enabled and self.ou_epsilon > 0:
            new_a_states = []
            for a_state, action in zip(agent_states, actions):
                if a_state is None:
                    a_state = np.zeros(
                        shape=action.shape, dtype=np.float32)
                a_state += self.ou_teta * (self.ou_mu - a_state)
                a_state += self.ou_sigma * np.random.normal(
                    size=action.shape)

                action += self.ou_epsilon * a_state
                new_a_states.append(a_state)
        else:
            new_a_states = agent_states

        actions = np.clip(actions, -self.threshold, self.threshold)
        return actions, new_a_states


def DDPG_TRAIN(env: gym.Env,
               gamma=0.99,
               batch_size=64,
               lr=1e-4,
               replay_size=100000,
               replay_initial=1000,
               frame_upper_bound=10000,
               threshold=2.0,
               continue_train_flag=False,
               continue_train_act_para=None,
               continue_train_crt_para=None):
    time_start = time.time()
    save_path = os.path.join("saves", "ddpg" + str(time_start))
    os.makedirs(save_path, exist_ok=True)
    writer = SummaryWriter(comment="-ddpg")

    if continue_train_flag:
        act_net = DDPGActor(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            threshold)
        crt_net = DDPGCritic(
            env.observation_space.shape[0],
            env.action_space.shape[0])
        act_net.load_state_dict(continue_train_act_para)
        crt_net.load_state_dict(continue_train_crt_para)
        act_net = act_net.to(device)
        crt_net = crt_net.to(device)
    else:
        act_net = DDPGActor(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            threshold).to(device)
        crt_net = DDPGCritic(
            env.observation_space.shape[0],
            env.action_space.shape[0]).to(device)
    tgt_act_net = ptan.agent.TargetNet(act_net)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)

    agent = AgentDDPG(act_net, device=device, threshold=threshold)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=gamma, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=replay_size)
    act_opt = optim.Adam(act_net.parameters(), lr=lr)
    crt_opt = optim.Adam(crt_net.parameters(), lr=lr)

    frame_idx = 0
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(
                writer, batch_size=10) as tb_tracker:
            while True:
                frame_idx += 1
                if frame_idx > frame_upper_bound:
                    break
                buffer.populate(1)
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], frame_idx)
                    tracker.reward(rewards[0], frame_idx)

                if len(buffer) < replay_initial:
                    continue

                batch = buffer.sample(batch_size)
                states_v, actions_v, rewards_v, \
                dones_mask, last_states_v = \
                    unpack_batch_ddqn(batch, device)

                # train critic
                crt_opt.zero_grad()
                q_v = crt_net(states_v, actions_v)
                last_act_v = tgt_act_net.target_model(
                    last_states_v)
                q_last_v = tgt_crt_net.target_model(
                    last_states_v, last_act_v)
                q_last_v[dones_mask] = 0.0
                q_ref_v = rewards_v.unsqueeze(dim=-1) + \
                          q_last_v * gamma
                critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                critic_loss_v.backward()
                crt_opt.step()
                tb_tracker.track("loss_critic",
                                 critic_loss_v, frame_idx)
                tb_tracker.track("critic_ref",
                                 q_ref_v.mean(), frame_idx)

                # train actor
                act_opt.zero_grad()
                cur_actions_v = act_net(states_v)
                actor_loss_v = -crt_net(states_v, cur_actions_v)
                actor_loss_v = actor_loss_v.mean()
                actor_loss_v.backward()
                act_opt.step()
                tb_tracker.track("loss_actor",
                                 actor_loss_v, frame_idx)

                tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

    # You need to reconstruct buffer using buffer.buffer
    return act_net, crt_net, buffer.buffer
