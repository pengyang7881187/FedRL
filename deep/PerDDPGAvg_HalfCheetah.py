import os
import gym
import ptan
import time
import copy
import math
import torch
import numpy as np
import argparse
import matplotlib as mpl
mpl.use('Agg')
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from DDPGAvg import net_para_add, net_para_scale
from DeepRLAlgo import DDPG_TRAIN as M_DDPG_TRAIN
from DeepRLAlgo import DDPGActor as M_DDPGActor
from DeepRLAlgo import DDPGCritic as M_DDPGCritic
from DeepRLAlgo import RLNet, AgentDDPG, unpack_batch_ddqn
from Per_ExpRecorder import AbsoluteRewardRecorder as Per_AbsoluteRewardRecorder
from ExpRecorder import AbsoluteRewardRecorder as Ori_AbsoluteRewardRecorder
from typing import Type
from Per_MyHalfCheetah import MyHalfCheetahEnv as Per_MyHalfCheetahEnv
from MyHalfCheetah import MyHalfCheetahEnv as Ori_MyHalfCheetahEnv
from tqdm import tqdm
from tensorboardX import SummaryWriter
from Config import device, MERGE_INTERVAL, GAMMA, EVALUATE_EPISODES_FOR_EVAL
from MyEnv import MyEnv

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--lr_embedding", default=1e-4, type=float)
parser.add_argument("--train_size", default=10, type=int)
parser.add_argument("--validation_size", default=10, type=int)
parser.add_argument("--embed_type", default="Gauss", type=str)
parser.add_argument("--outer_loop_num", default=100, type=int)
parser.add_argument("--continue_ub", default=80000, type=int)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--seed", default=3, type=int)
args = parser.parse_args()

LR_PER, LR_EMBED = args.lr, args.lr_embedding
TRAIN_SIZE_PER, VALIDATION_SIZE_PER = args.train_size, args.validation_size
EMBED_TYPE = args.embed_type
OUTER_LOOP_NUM = args.outer_loop_num
CONTINUE_TRAIN_UB = args.continue_ub
GPU = args.gpu
seed = args.seed

EVALUATE_TIME_FOR_PER = 50

os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)

save_dir_name = 'plot-PerDDPGAvg-HalfCheetah'
global_log_dir = './PerDDPGAvg-HalfCheetah/'

threshold = 1.0  # This parameter is only for HalfCheetah

outer_loop_num = OUTER_LOOP_NUM


@torch.no_grad()
def Per_evaluate_expected_reward(net, env, evaluate_episodes_for_eval):
    # For DPG-network only
    evaluate_rewards = 0.0  # Used for evaluate model's performance

    for _ in range(evaluate_episodes_for_eval):
        obs = env.reset()
        steps = 0
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)
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


class Per_DDPGActor(RLNet):
    def __init__(self, obs_size, act_size, dim_embedding, threshold=2.0):
        super(Per_DDPGActor, self).__init__(obs_size=obs_size, action_size=act_size)
        self.network_type = 'DP'
        self.rescale = threshold

        self.net = nn.Sequential(
            nn.Linear(obs_size + dim_embedding, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh()
        )

    def forward(self, x_em):
        return self.rescale * self.net(x_em.float())


class Per_DDPGCritic(nn.Module):
    def __init__(self, obs_size, act_size, dim_embedding):
        super(Per_DDPGCritic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size + dim_embedding, 400),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, x_em, a):
        obs = self.obs_net(x_em.float())
        return self.out_net(torch.cat([obs, a], dim=1))


class Per_AgentDDPG(ptan.agent.BaseAgent):
    """
    Agent implementing Orstein-Uhlenbeck exploration process
    """
    def __init__(self, net, device="cpu", ou_enabled=True,
                 ou_mu=0.0, ou_teta=0.15, ou_sigma=0.2,
                 ou_epsilon=1.0, threshold=2.0, preprocessor=None):
        self.net = net
        self.device = device
        self.threshold = threshold
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_teta = ou_teta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon
        self.preprocessor = preprocessor

    def initial_state(self):
        return None

    def __call__(self, states, agent_states):
        if self.preprocessor is not None:
            states_v = self.preprocessor(states)
        else:
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


@torch.no_grad()
def Per_unpack_batch_ddqn(batch, device="cpu", update_embedding=False, embedding=None):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = exp.state
        if update_embedding:
            dim_embedding = embedding.size
            state[:dim_embedding] = embedding
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_state = exp.state
            if update_embedding:
                dim_embedding = embedding.size
                last_state[:dim_embedding] = embedding
            last_states.append(last_state)
        else:
            last_state = exp.last_state
            if update_embedding:
                dim_embedding = embedding.size
                last_state[:dim_embedding] = embedding
            last_states.append(last_state)
    states_v = ptan.agent.float32_preprocessor(states).to(device)
    actions_v = ptan.agent.float32_preprocessor(actions).to(device)
    rewards_v = ptan.agent.float32_preprocessor(rewards).to(device)
    last_states_v = ptan.agent.float32_preprocessor(last_states).to(device)
    dones_t = torch.BoolTensor(dones).to(device)
    return states_v, actions_v, rewards_v, dones_t, last_states_v


def init_embedding(train_size, validation_size, mode):
    assert mode in ['one-hot', 'Gauss']
    if mode == 'one-hot':
        embedding_dim = train_size
        train_embeddings = np.identity(embedding_dim)
        validation_embeddings = np.random.uniform(0, 1, (validation_size, embedding_dim))
        validation_normalizing = np.sum(validation_embeddings, axis=1, keepdims=True)
        validation_embeddings = validation_embeddings / validation_normalizing
    else:
        embedding_dim = math.ceil(np.log2(train_size))
        train_embeddings = np.random.uniform(-1, 1, (train_size, embedding_dim))
        validation_embeddings = np.random.uniform(-1, 1, (validation_size, embedding_dim))
    return embedding_dim, train_embeddings, validation_embeddings


def Per_DDPG_TRAIN(env: gym.Env,
                   act_embedding,
                   crt_embedding,
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

    dim_embedding = act_embedding.size
    env.embedding = act_embedding

    act_embedding_tensor = (torch.tensor(act_embedding)).to(device)
    act_embedding_tensor.requires_grad_(True)
    crt_embedding_tensor = (torch.tensor(crt_embedding)).to(device)
    crt_embedding_tensor.requires_grad_(True)

    if continue_train_flag:
        act_net = Per_DDPGActor(
            obs_size=Ori_MyHalfCheetahEnv().observation_space.shape[0],
            act_size=env.action_space.shape[0],
            dim_embedding=dim_embedding,
            threshold=threshold)
        crt_net = Per_DDPGCritic(
            obs_size=Ori_MyHalfCheetahEnv().observation_space.shape[0],
            act_size=env.action_space.shape[0],
            dim_embedding=dim_embedding)
        act_net.load_state_dict(continue_train_act_para)
        crt_net.load_state_dict(continue_train_crt_para)
        act_net = act_net.to(device)
        crt_net = crt_net.to(device)
    else:
        act_net = Per_DDPGActor(
            obs_size=Ori_MyHalfCheetahEnv().observation_space.shape[0],
            act_size=env.action_space.shape[0],
            dim_embedding=dim_embedding,
            threshold=threshold).to(device)
        crt_net = Per_DDPGCritic(
            obs_size=Ori_MyHalfCheetahEnv().observation_space.shape[0],
            act_size=env.action_space.shape[0],
            dim_embedding=dim_embedding).to(device)
    tgt_act_net = ptan.agent.TargetNet(act_net)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)

    @torch.no_grad()
    def state_preprocessor_act_clone(states):
        act_embedding_tensor_clone = act_embedding_tensor.clone().detach()
        if torch.is_tensor(states):
            t_states = (states.clone().detach()).to(device)
        else:
            t_states = (torch.tensor(states)).to(device)
        if t_states.ndim == 1 or t_states.shape[1] == 1:
            t_states[:dim_embedding] = act_embedding_tensor_clone
        else:
            t_states[:, :dim_embedding] = act_embedding_tensor_clone
        return t_states

    agent = Per_AgentDDPG(act_net, device=device,
                          threshold=threshold, preprocessor=state_preprocessor_act_clone)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=gamma, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=replay_size)

    act_parameters = [
        {'params': act_net.parameters()},
        {'params': [act_embedding_tensor]}
    ]
    crt_parameters = [
        {'params': crt_net.parameters()},
        {'params': [crt_embedding_tensor]}
    ]
    act_opt = optim.Adam(act_parameters, lr=lr)
    crt_opt = optim.Adam(crt_parameters, lr=lr)

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
                    Per_unpack_batch_ddqn(batch=batch, device=device, update_embedding=True, embedding=act_embedding.copy())

                def state_preprocessor_for_act_tensor(t_states):
                    if t_states.ndim == 1 or t_states.shape[1] == 1:
                        t_states[:dim_embedding] = act_embedding_tensor
                    else:
                        t_states[:, :dim_embedding] = act_embedding_tensor
                    return t_states

                def state_preprocessor_for_crt_tensor(t_states):
                    if t_states.ndim == 1 or t_states.shape[1] == 1:
                        t_states[:dim_embedding] = crt_embedding_tensor
                    else:
                        t_states[:, :dim_embedding] = crt_embedding_tensor
                    return t_states

                @torch.no_grad()
                def state_preprocessor_crt_clone(states):
                    crt_embedding_tensor_clone = crt_embedding_tensor.clone().detach()
                    if torch.is_tensor(states):
                        t_states = (states.clone().detach()).to(device)
                    else:
                        t_states = (torch.tensor(states)).to(device)
                    if t_states.ndim == 1 or t_states.shape[1] == 1:
                        t_states[:dim_embedding] = crt_embedding_tensor_clone
                    else:
                        t_states[:, :dim_embedding] = crt_embedding_tensor_clone
                    return t_states

                # train critic
                crt_opt.zero_grad()
                states_v_crt = state_preprocessor_for_crt_tensor(states_v)
                q_v = crt_net(states_v_crt, actions_v)

                last_states_v_act_clone = state_preprocessor_act_clone(last_states_v)
                last_states_v_crt_clone = state_preprocessor_crt_clone(last_states_v)

                last_act_v = tgt_act_net.target_model(
                    last_states_v_act_clone)
                q_last_v = tgt_crt_net.target_model(
                    last_states_v_crt_clone, last_act_v)
                q_last_v[dones_mask] = 0.0
                q_ref_v = rewards_v.unsqueeze(dim=-1) + \
                          q_last_v * gamma
                critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                critic_loss_v.backward(retain_graph=True)
                crt_opt.step()
                tb_tracker.track("loss_critic",
                                 critic_loss_v, frame_idx)
                tb_tracker.track("critic_ref",
                                 q_ref_v.mean(), frame_idx)

                # train actor
                act_opt.zero_grad()
                states_v_act = state_preprocessor_for_act_tensor(states_v)
                cur_actions_v = act_net(states_v_act)

                states_v_crt_clone = state_preprocessor_crt_clone(states_v)

                actor_loss_v = -crt_net(states_v_crt_clone, cur_actions_v)
                actor_loss_v = actor_loss_v.mean()
                actor_loss_v.backward()
                act_opt.step()
                tb_tracker.track("loss_actor",
                                 actor_loss_v, frame_idx)

                tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

                act_embedding = act_embedding_tensor.clone().cpu().detach().numpy()
                crt_embedding = crt_embedding_tensor.clone().cpu().detach().numpy()

    # You need to reconstruct buffer using buffer.buffer
    return act_net, crt_net, act_embedding, crt_embedding, buffer.buffer


# 应该已改完
def M_DDPGAvg(env_class: Type[MyEnv],
            recorder_extra_hyperparameters: dict,  # Do not parse None to it!
            train_theta_set: np.ndarray,
            validation_theta_set: np.ndarray,
            agent_trainer=M_DDPG_TRAIN,
            gamma=0.99,
            merge_interval=MERGE_INTERVAL,
            total_frame_upper_bound=int(1e+6),
            log_dir='./log-pendulum/',
            crt_flag=True,
            silent_flag=True,
            threshold=2.0):

    recorder_class = Ori_AbsoluteRewardRecorder

    # mkdir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    os.makedirs(log_dir + 'train_M/')
    os.makedirs(log_dir + 'validation_M/')

    assert total_frame_upper_bound % merge_interval == 0  # This assumption is for convenience

    # Prepare environments and set some parameters
    # We do not need validation_size and test_size in this function
    train_size = train_theta_set.size
    env_train = []
    for theta in train_theta_set:
        env_train.append(env_class(para=theta))

    # Prepare recorder
    # Outer loop: train agents and merge their parameters
    outer_loop_iter_upper_bound = int(total_frame_upper_bound / merge_interval)
    train_recorder_log_dir = log_dir + 'train_M/'
    validation_recorder_log_dir = log_dir + 'validation_M/'
    train_recorder_name = 'Train recorder'
    validation_recorder_name = 'Validation recorder'
    record_len = outer_loop_iter_upper_bound  # We simply record every outer loop
    train_recorder = recorder_class(env_class=env_class,
                                    theta_set=train_theta_set,
                                    merge_interval=merge_interval,
                                    log_folder_dir=train_recorder_log_dir,
                                    record_len=record_len,
                                    recorder_name=train_recorder_name,
                                    gamma=gamma,
                                    **recorder_extra_hyperparameters)
    validation_recorder = recorder_class(env_class=env_class,
                                         theta_set=validation_theta_set,
                                         merge_interval=merge_interval,
                                         log_folder_dir=validation_recorder_log_dir,
                                         record_len=record_len,
                                         recorder_name=validation_recorder_name,
                                         gamma=gamma,
                                         **recorder_extra_hyperparameters)

    # Initialize network or load network parameter
    act_net = M_DDPGActor(
        env_train[0].observation_space.shape[0],
        env_train[0].action_space.shape[0],
        threshold).to(device)
    crt_net = M_DDPGCritic(
        env_train[0].observation_space.shape[0],
        env_train[0].action_space.shape[0]).to(device)

    crt_para_set = []

    # Outer loop: train agents and merge their parameters
    for i in range(outer_loop_iter_upper_bound):
        #########
        # Train #
        #########
        act_net_para = 0  # This is only for avoiding warning
        crt_net_para = 0  # This is only for avoiding warning

        # First round of iteration when continue flag is False
        if i == 0:
            for j in range(train_size):
                tmp_act_net, tmp_crt_net, _ = agent_trainer(env=env_train[j], frame_upper_bound=merge_interval,
                                                            threshold=threshold)
                tmp_act_para = tmp_act_net.state_dict()
                tmp_crt_para = tmp_crt_net.state_dict()
                crt_para_set.append(copy.deepcopy(tmp_crt_para))
                if j == 0:
                    act_net_para = copy.deepcopy(tmp_act_para)
                    if crt_flag:
                        crt_net_para = copy.deepcopy(tmp_crt_para)
                else:
                    act_net_para = net_para_add(act_net_para, tmp_act_para)
                    if crt_flag:
                        crt_net_para = net_para_add(crt_net_para, tmp_crt_para)

        # Other rounds of iteration, continue train from net
        else:
            for j in range(train_size):
                if crt_flag:
                    tmp_crt_para = copy.deepcopy(crt_net.state_dict())
                else:
                    tmp_crt_para = copy.deepcopy(crt_para_set[j])

                tmp_act_net, tmp_crt_net, _ = agent_trainer(env=env_train[j],
                                                            threshold=threshold,
                                                            frame_upper_bound=merge_interval,
                                                            continue_train_flag=True,
                                                            continue_train_act_para=act_net.state_dict(),
                                                            continue_train_crt_para=tmp_crt_para)
                tmp_act_para = tmp_act_net.state_dict()
                tmp_crt_para = tmp_crt_net.state_dict()
                crt_para_set[j] = copy.deepcopy(tmp_crt_para)
                if j == 0:
                    act_net_para = copy.deepcopy(tmp_act_para)
                    if crt_flag:
                        crt_net_para = copy.deepcopy(tmp_crt_para)
                else:
                    act_net_para = net_para_add(act_net_para, tmp_act_para)
                    if crt_flag:
                        crt_net_para = net_para_add(crt_net_para, tmp_crt_para)
        act_net_para = net_para_scale(act_net_para, (1 / train_size))
        act_net.load_state_dict(act_net_para)
        if crt_flag:
            crt_net_para = net_para_scale(crt_net_para, (1 / train_size))
            crt_net.load_state_dict(crt_net_para)

        ################
        # Record error #
        ################

        # Record training error
        train_recorder.record(net=act_net)
        # Record validation error
        validation_recorder.record(net=act_net)
        print('report')

        if not silent_flag:
            # Report error
            train_recorder.report()
            validation_recorder.report()
    # End outer loop here

    return act_net, crt_net, train_recorder, validation_recorder


# 应该已改完
def Per_DDPGAvg(env_class: Type[MyEnv],
            recorder_extra_hyperparameters: dict,  # Do not parse None to it!
            train_theta_set: np.ndarray,
            validation_theta_set: np.ndarray,
            dim_embedding,
            act_train_embeddings,
            act_validation_embeddings,
            crt_train_embeddings,
            agent_trainer=Per_DDPG_TRAIN,
            gamma=0.99,
            merge_interval=MERGE_INTERVAL,
            total_frame_upper_bound=int(1e+6),
            log_dir='./log-pendulum/',
            crt_flag=True,
            silent_flag=True,
            threshold=2.0):

    recorder_class = Per_AbsoluteRewardRecorder

    # mkdir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    os.makedirs(log_dir + 'train/')
    os.makedirs(log_dir + 'validation/')

    assert total_frame_upper_bound % merge_interval == 0  # This assumption is for convenience

    # Prepare environments and set some parameters
    # We do not need validation_size and test_size in this function
    train_size = train_theta_set.size
    env_train = []
    for theta in train_theta_set:
        env_train.append(env_class(para=theta))

    # Prepare recorder
    # Outer loop: train agents and merge their parameters
    outer_loop_iter_upper_bound = int(total_frame_upper_bound / merge_interval)
    train_recorder_log_dir = log_dir + 'train/'
    validation_recorder_log_dir = log_dir + 'validation/'
    train_recorder_name = 'Train recorder'
    validation_recorder_name = 'Validation recorder'
    record_len = outer_loop_iter_upper_bound  # We simply record every outer loop
    train_recorder = recorder_class(env_class=env_class,
                                    theta_set=train_theta_set,
                                    merge_interval=merge_interval,
                                    log_folder_dir=train_recorder_log_dir,
                                    record_len=record_len,
                                    recorder_name=train_recorder_name,
                                    gamma=gamma,
                                    embedding_set=act_train_embeddings,
                                    **recorder_extra_hyperparameters)
    validation_recorder = recorder_class(env_class=env_class,
                                         theta_set=validation_theta_set,
                                         merge_interval=merge_interval,
                                         log_folder_dir=validation_recorder_log_dir,
                                         record_len=record_len,
                                         recorder_name=validation_recorder_name,
                                         gamma=gamma,
                                         embedding_set=act_validation_embeddings,
                                         **recorder_extra_hyperparameters)

    # Initialize network or load network parameter
    act_net = Per_DDPGActor(
        obs_size=Ori_MyHalfCheetahEnv().observation_space.shape[0],
        act_size=env_train[0].action_space.shape[0],
        dim_embedding=dim_embedding,
        threshold=threshold).to(device)
    crt_net = Per_DDPGCritic(
        obs_size=Ori_MyHalfCheetahEnv().observation_space.shape[0],
        act_size=env_train[0].action_space.shape[0],
        dim_embedding=dim_embedding).to(device)

    crt_para_set = []

    act_embeddings = act_train_embeddings.copy()

    crt_embeddings = crt_train_embeddings.copy()

    # Outer loop: train agents and merge their parameters
    for i in range(outer_loop_iter_upper_bound):
        #########
        # Train #
        #########
        act_net_para = 0  # This is only for avoiding warning
        crt_net_para = 0  # This is only for avoiding warning

        # First round of iteration when continue flag is False
        if i == 0:
            for j in range(train_size):
                tmp_act_net, tmp_crt_net, tmp_act_embedding, tmp_crt_embedding, _ = \
                    agent_trainer(env=env_train[j], frame_upper_bound=merge_interval,
                                  threshold=threshold, act_embedding=act_embeddings[j],
                                  crt_embedding=crt_embeddings[j])
                tmp_act_para = tmp_act_net.state_dict()
                tmp_crt_para = tmp_crt_net.state_dict()
                act_embeddings[j] = tmp_act_embedding.copy()
                crt_embeddings[j] = tmp_crt_embedding.copy()
                crt_para_set.append(copy.deepcopy(tmp_crt_para))
                if j == 0:
                    act_net_para = copy.deepcopy(tmp_act_para)
                    if crt_flag:
                        crt_net_para = copy.deepcopy(tmp_crt_para)
                else:
                    act_net_para = net_para_add(act_net_para, tmp_act_para)
                    if crt_flag:
                        crt_net_para = net_para_add(crt_net_para, tmp_crt_para)

        # Other rounds of iteration, continue train from net
        else:
            for j in range(train_size):
                # print('ok1')
                if crt_flag:
                    tmp_crt_para = copy.deepcopy(crt_net.state_dict())
                else:
                    tmp_crt_para = copy.deepcopy(crt_para_set[j])

                tmp_act_net, tmp_crt_net, tmp_act_embedding, tmp_crt_embedding, _ = \
                    agent_trainer(env=env_train[j],
                                  threshold=threshold,
                                  act_embedding=act_embeddings[j],
                                  crt_embedding=crt_embeddings[j],
                                  frame_upper_bound=merge_interval,
                                  continue_train_flag=True,
                                  continue_train_act_para=act_net.state_dict(),
                                  continue_train_crt_para=tmp_crt_para)
                tmp_act_para = tmp_act_net.state_dict()
                tmp_crt_para = tmp_crt_net.state_dict()
                act_embeddings[j] = tmp_act_embedding.copy()
                crt_embeddings[j] = tmp_crt_embedding.copy()
                crt_para_set[j] = copy.deepcopy(tmp_crt_para)
                if j == 0:
                    act_net_para = copy.deepcopy(tmp_act_para)
                    if crt_flag:
                        crt_net_para = copy.deepcopy(tmp_crt_para)
                else:
                    act_net_para = net_para_add(act_net_para, tmp_act_para)
                    if crt_flag:
                        crt_net_para = net_para_add(crt_net_para, tmp_crt_para)
        act_net_para = net_para_scale(act_net_para, (1 / train_size))
        act_net.load_state_dict(act_net_para)
        if crt_flag:
            crt_net_para = net_para_scale(crt_net_para, (1 / train_size))
            crt_net.load_state_dict(crt_net_para)

        ################
        # Record error #
        ################

        train_recorder.embedding_set = act_embeddings.copy()

        # Record training error
        train_recorder.record(net=act_net)
        # Record validation error
        validation_recorder.record(net=act_net)
        print('report')

        if not silent_flag:
            # Report error
            train_recorder.report()
            validation_recorder.report()
    # End outer loop here
    return act_net, crt_net, act_embeddings, crt_embeddings, train_recorder, validation_recorder


# We separate train and report, cuz train is very time-consuming, while report is not
def integrate_diff_merge_interval():
    # mkdir
    log_dir = global_log_dir

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists('./' + save_dir_name):
        os.makedirs('./' + save_dir_name)

    np.random.seed(seed)

    # Prepare some parameters: most are loaded from Config.py
    env_class = Per_MyHalfCheetahEnv
    gamma = 0.99

    total_frame_upper_bound = int(320000)

    train_size = TRAIN_SIZE_PER
    validation_size = VALIDATION_SIZE_PER

    embed_mode = EMBED_TYPE

    train_theta_set = np.random.uniform(size=(outer_loop_num, train_size))
    validation_theta_set = np.linspace(0, 1, validation_size)
    np.save(log_dir + 'train_theta.npy', train_theta_set)
    np.save(log_dir + 'validation_theta.npy', validation_theta_set)

    dim_embedding, _, _ = \
        init_embedding(train_size, validation_size, mode=embed_mode)

    act_total_train_embeddings = np.zeros((outer_loop_num, train_size, dim_embedding))
    act_total_validation_embeddings = np.zeros((outer_loop_num, validation_size, dim_embedding))
    for i in range(outer_loop_num):
        _, act_train_embeddings, act_validation_embeddings = \
            init_embedding(train_size, validation_size, mode=embed_mode)
        act_total_train_embeddings[i] = act_train_embeddings.copy()
        act_total_validation_embeddings[i] = act_validation_embeddings.copy()
    np.save(log_dir + 'act_total_train_embedding.npy', act_total_train_embeddings)
    np.save(log_dir + 'act_total_validation_embedding.npy', act_total_validation_embeddings)

    crt_total_train_embeddings = np.zeros((outer_loop_num, train_size, dim_embedding))
    crt_total_validation_embeddings = np.zeros((outer_loop_num, validation_size, dim_embedding))
    for i in range(outer_loop_num):
        _, crt_train_embeddings, crt_validation_embeddings = \
            init_embedding(train_size, validation_size, mode=embed_mode)
        crt_total_train_embeddings[i] = crt_train_embeddings.copy()
        crt_total_validation_embeddings[i] = crt_validation_embeddings.copy()
    np.save(log_dir + 'crt_total_train_embedding.npy', crt_total_train_embeddings)
    np.save(log_dir + 'crt_total_validation_embedding.npy', crt_total_validation_embeddings)

    agent_trainer = Per_DDPG_TRAIN

    evaluate_episodes_for_eval = EVALUATE_EPISODES_FOR_EVAL
    recorder_extra_hyperparameters = {'evaluate_episodes_for_eval': evaluate_episodes_for_eval}

    merge_interval_set = [20000, 40000]

    exp_diff_merge_interval(env_class=env_class,
                            agent_trainer=agent_trainer,
                            act_total_train_embeddings=act_total_train_embeddings,
                            act_total_validation_embeddings=act_total_validation_embeddings,
                            crt_total_train_embeddings=crt_total_train_embeddings,
                            crt_total_validation_embeddings=crt_total_validation_embeddings,
                            recorder_extra_hyperparameters=recorder_extra_hyperparameters,
                            merge_interval_set=merge_interval_set,
                            gamma=gamma,
                            total_frame_upper_bound=total_frame_upper_bound,
                            log_dir=log_dir,
                            dim_embedding=dim_embedding)
    report_diff_merge_interval(total_frame_upper_bound=total_frame_upper_bound,
                               merge_interval_set=merge_interval_set)
    return


def integrate_personalization(
        Act_net: Per_DDPGActor,
        Crt_net: Per_DDPGCritic,
        M_act_net: M_DDPGActor,
        M_crt_net: M_DDPGCritic,
        dim_embedding,
        act_validation_embeddings,
        crt_validation_embeddings,
        validation_theta_set,
        gamma,
        log_dir,
        threshold,
        init_embedding_flag=False,
        act_init_embeddings=None,
        crt_init_embeddings=None):
    env_class = Per_MyHalfCheetahEnv
    ori_env_class = Ori_MyHalfCheetahEnv

    current_log_dir = log_dir + 'person/'
    if not os.path.exists(current_log_dir):
        os.makedirs(current_log_dir)

    validation_size = validation_theta_set.size

    act_validation_embeddings_for_use = act_validation_embeddings.copy()
    crt_validation_embeddings_for_use = crt_validation_embeddings.copy()

    if init_embedding_flag:
        act_avg_embeddings = np.average(act_init_embeddings, axis=0)
        crt_avg_embeddings = np.average(crt_init_embeddings, axis=0)
        for ttt in range(validation_size):
            act_validation_embeddings_for_use[ttt] = act_avg_embeddings.copy()
            crt_validation_embeddings_for_use[ttt] = crt_avg_embeddings.copy()

    env_validation = []
    for theta in validation_theta_set:
        env_validation.append(env_class(para=theta))
    obs_size = Ori_MyHalfCheetahEnv().observation_space.shape[0]
    action_size = env_validation[0].action_space.shape[0]

    def exp_personalization(ori_act_net, ori_crt_net, type):
        act_ori_para = ori_act_net.state_dict()
        crt_ori_para = ori_crt_net.state_dict()

        performance = np.zeros((validation_size, EVALUATE_TIME_FOR_PER))

        evaluate_interval = CONTINUE_TRAIN_UB / EVALUATE_TIME_FOR_PER

        act_current_validation_embeddings = act_validation_embeddings_for_use.copy()
        crt_current_validation_embeddings = crt_validation_embeddings_for_use.copy()

        for i in range(validation_size):
            env = env_class(para=validation_theta_set[i])
            act_embedding = copy.deepcopy(act_current_validation_embeddings[i])
            crt_embedding = copy.deepcopy(crt_current_validation_embeddings[i])

            env.embedding = act_embedding

            act_embedding_tensor = (torch.tensor(act_embedding)).to(device)
            crt_embedding_tensor = (torch.tensor(crt_embedding)).to(device)
            act_embedding_tensor.requires_grad_(True)
            crt_embedding_tensor.requires_grad_(True)

            act_net_exp_para = copy.deepcopy(act_ori_para)
            crt_net_exp_para = copy.deepcopy(crt_ori_para)
            act_net_exp = Per_DDPGActor(
                obs_size=obs_size,
                act_size=action_size,
                dim_embedding=dim_embedding,
                threshold=threshold)
            crt_net_exp = Per_DDPGCritic(
                obs_size=obs_size,
                act_size=action_size,
                dim_embedding=dim_embedding)
            act_net_exp.load_state_dict(act_net_exp_para)
            crt_net_exp.load_state_dict(crt_net_exp_para)
            act_net_exp = act_net_exp.to(device)
            crt_net_exp = crt_net_exp.to(device)

            tgt_act_net = ptan.agent.TargetNet(act_net_exp)
            tgt_crt_net = ptan.agent.TargetNet(crt_net_exp)

            time_start = time.time()
            save_path = os.path.join("saves", "ddpg" + str(time_start))
            os.makedirs(save_path, exist_ok=True)
            writer = SummaryWriter(comment="-ddpg")

            batch_size = 64
            lr = LR_PER
            lr_embedding = LR_EMBED
            replay_size = 10000
            replay_initial = 200

            def state_preprocessor_for_act_tensor(t_states):
                if t_states.ndim == 1 or t_states.shape[1] == 1:
                    t_states[:dim_embedding] = act_embedding_tensor
                else:
                    t_states[:, :dim_embedding] = act_embedding_tensor
                return t_states

            def state_preprocessor_for_crt_tensor(t_states):
                if t_states.ndim == 1 or t_states.shape[1] == 1:
                    t_states[:dim_embedding] = crt_embedding_tensor
                else:
                    t_states[:, :dim_embedding] = crt_embedding_tensor
                return t_states

            @torch.no_grad()
            def state_preprocessor_act_clone(states):
                act_embedding_tensor_clone = act_embedding_tensor.clone().detach()
                if torch.is_tensor(states):
                    t_states = (states.clone().detach()).to(device)
                else:
                    t_states = (torch.tensor(states)).to(device)
                if t_states.ndim == 1 or t_states.shape[1] == 1:
                    t_states[:dim_embedding] = act_embedding_tensor_clone
                else:
                    t_states[:, :dim_embedding] = act_embedding_tensor_clone
                return t_states

            @torch.no_grad()
            def state_preprocessor_crt_clone(states):
                crt_embedding_tensor_clone = crt_embedding_tensor.clone().detach()
                if torch.is_tensor(states):
                    t_states = (states.clone().detach()).to(device)
                else:
                    t_states = (torch.tensor(states)).to(device)
                if t_states.ndim == 1 or t_states.shape[1] == 1:
                    t_states[:dim_embedding] = crt_embedding_tensor_clone
                else:
                    t_states[:, :dim_embedding] = crt_embedding_tensor_clone
                return t_states

            agent = Per_AgentDDPG(act_net_exp, device=device,
                                  threshold=threshold, preprocessor=state_preprocessor_act_clone)
            exp_source = ptan.experience.ExperienceSourceFirstLast(
                env, agent, gamma=gamma, steps_count=1)
            buffer = ptan.experience.ExperienceReplayBuffer(
                exp_source, buffer_size=replay_size)

            if type == 'Exp-1':
                # Embedding only
                act_opt = optim.Adam([act_embedding_tensor], lr_embedding)
                crt_opt = optim.Adam([crt_embedding_tensor], lr_embedding)
            elif type == 'Exp-2':
                # Both embedding and net
                act_parameters = [
                    {'params': act_net_exp.parameters()},
                    {'params': [act_embedding_tensor], 'lr': lr_embedding}
                ]
                crt_parameters = [
                    {'params': crt_net_exp.parameters()},
                    {'params': [crt_embedding_tensor], 'lr': lr_embedding}
                ]
                act_opt = optim.Adam(act_parameters, lr=lr)
                crt_opt = optim.Adam(crt_parameters, lr=lr)
            elif type == 'Ctr-1':
                # Net only
                act_opt = optim.Adam(act_net_exp.parameters(), lr=lr)
                crt_opt = optim.Adam(crt_net_exp.parameters(), lr=lr)
            elif type == 'Ctr-2':
                # Not update
                act_opt = None
                crt_opt = None
            else:
                raise

            frame_idx = 0
            episode = 0

            with ptan.common.utils.RewardTracker(writer) as tracker:
                with ptan.common.utils.TBMeanTracker(
                        writer, batch_size=10) as tb_tracker:
                    while True:
                        frame_idx += 1
                        if frame_idx >= CONTINUE_TRAIN_UB:
                            break
                        if frame_idx % evaluate_interval == 0:
                            # Evaluate here
                            tmp_env = env_class(para=validation_theta_set[i])
                            tmp_env.embedding = act_embedding

                            performance[i][episode] = Per_evaluate_expected_reward(net=act_net_exp, env=tmp_env,
                                                                                   evaluate_episodes_for_eval=30)
                            episode += 1
                            if episode >= EVALUATE_TIME_FOR_PER:
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
                            Per_unpack_batch_ddqn(batch=batch, device=device, update_embedding=True,
                                                  embedding=act_embedding.copy())
                        # train critic
                        if type != 'Ctr-2':
                            crt_opt.zero_grad()
                        states_v_crt = state_preprocessor_for_crt_tensor(states_v)
                        q_v = crt_net_exp(states_v_crt, actions_v)

                        last_states_v_act_clone = state_preprocessor_act_clone(last_states_v)
                        last_states_v_crt_clone = state_preprocessor_crt_clone(last_states_v)

                        last_act_v = tgt_act_net.target_model(
                            last_states_v_act_clone)
                        q_last_v = tgt_crt_net.target_model(
                            last_states_v_crt_clone, last_act_v)
                        q_last_v[dones_mask] = 0.0
                        q_ref_v = rewards_v.unsqueeze(dim=-1) + \
                                  q_last_v * gamma
                        critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                        if type != 'Ctr-2':
                            critic_loss_v.backward(retain_graph=True)
                            crt_opt.step()
                        tb_tracker.track("loss_critic",
                                         critic_loss_v, frame_idx)
                        tb_tracker.track("critic_ref",
                                         q_ref_v.mean(), frame_idx)

                        # train actor
                        if type != 'Ctr-2':
                            act_opt.zero_grad()
                        states_v_act = state_preprocessor_for_act_tensor(states_v)
                        cur_actions_v = act_net_exp(states_v_act)

                        states_v_crt_clone = state_preprocessor_crt_clone(states_v)

                        actor_loss_v = -crt_net_exp(states_v_crt_clone, cur_actions_v)
                        actor_loss_v = actor_loss_v.mean()
                        if type != 'Ctr-2':
                            actor_loss_v.backward()
                            act_opt.step()
                        tb_tracker.track("loss_actor",
                                         actor_loss_v, frame_idx)

                        tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                        tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

                        act_embedding = act_embedding_tensor.clone().cpu().detach().numpy()
        return performance

    def ori_personalization(ori_act_net, ori_crt_net):
        act_ori_para = ori_act_net.state_dict()
        crt_ori_para = ori_crt_net.state_dict()

        performance = np.zeros((validation_size, EVALUATE_TIME_FOR_PER))

        evaluate_interval = CONTINUE_TRAIN_UB / EVALUATE_TIME_FOR_PER

        for i in range(validation_size):
            env = ori_env_class(para=validation_theta_set[i])

            act_net_exp_para = copy.deepcopy(act_ori_para)
            crt_net_exp_para = copy.deepcopy(crt_ori_para)
            act_net_exp = M_DDPGActor(
                obs_size=obs_size,
                act_size=action_size,
                threshold=threshold)
            crt_net_exp = M_DDPGCritic(
                obs_size=obs_size,
                act_size=action_size)
            act_net_exp.load_state_dict(act_net_exp_para)
            crt_net_exp.load_state_dict(crt_net_exp_para)
            act_net_exp = act_net_exp.to(device)
            crt_net_exp = crt_net_exp.to(device)

            tgt_act_net = ptan.agent.TargetNet(act_net_exp)
            tgt_crt_net = ptan.agent.TargetNet(crt_net_exp)

            time_start = time.time()
            save_path = os.path.join("saves", "ddpg" + str(time_start))
            os.makedirs(save_path, exist_ok=True)
            writer = SummaryWriter(comment="-ddpg")

            batch_size = 64
            lr = LR_PER
            replay_size = 10000
            replay_initial = 200

            agent = AgentDDPG(act_net_exp, device=device, threshold=threshold)
            exp_source = ptan.experience.ExperienceSourceFirstLast(
                env, agent, gamma=gamma, steps_count=1)
            buffer = ptan.experience.ExperienceReplayBuffer(
                exp_source, buffer_size=replay_size)
            act_opt = optim.Adam(act_net_exp.parameters(), lr=lr)
            crt_opt = optim.Adam(crt_net_exp.parameters(), lr=lr)

            frame_idx = 0
            episode = 0

            with ptan.common.utils.RewardTracker(writer) as tracker:
                with ptan.common.utils.TBMeanTracker(
                        writer, batch_size=10) as tb_tracker:
                    while True:
                        frame_idx += 1
                        if frame_idx >= CONTINUE_TRAIN_UB:
                            break
                        if frame_idx % evaluate_interval == 0:
                            # Evaluate here
                            tmp_env = ori_env_class(para=validation_theta_set[i])
                            performance[i][episode] = Per_evaluate_expected_reward(net=act_net_exp, env=tmp_env,
                                                                                   evaluate_episodes_for_eval=30)
                            episode += 1
                            if episode >= EVALUATE_TIME_FOR_PER:
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
                        q_v = crt_net_exp(states_v, actions_v)
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
                        cur_actions_v = act_net_exp(states_v)
                        actor_loss_v = -crt_net_exp(states_v, cur_actions_v)
                        actor_loss_v = actor_loss_v.mean()
                        actor_loss_v.backward()
                        act_opt.step()
                        tb_tracker.track("loss_actor",
                                         actor_loss_v, frame_idx)

                        tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                        tgt_crt_net.alpha_sync(alpha=1 - 1e-3)
        return performance

    # Numpy array of shape (validation_size, CONTINUE_TRAIN_UB)
    exp1_performance = exp_personalization(Act_net, Crt_net, 'Exp-1')
    exp2_performance = exp_personalization(Act_net, Crt_net, 'Exp-2')
    ctr1_performance = exp_personalization(Act_net, Crt_net, 'Ctr-1')
    ctr2_performance = exp_personalization(Act_net, Crt_net, 'Ctr-2')
    ctr3_performance = ori_personalization(M_act_net, M_crt_net)

    np.save(current_log_dir + 'Exp1.npy', exp1_performance)
    np.save(current_log_dir + 'Exp2.npy', exp2_performance)
    np.save(current_log_dir + 'Ctr1.npy', ctr1_performance)
    np.save(current_log_dir + 'Ctr2.npy', ctr2_performance)
    np.save(current_log_dir + 'Ctr3.npy', ctr3_performance)

    avg_exp1_performance = np.average(exp1_performance, axis=0)
    avg_exp2_performance = np.average(exp2_performance, axis=0)
    avg_ctr1_performance = np.average(ctr1_performance, axis=0)
    avg_ctr2_performance = np.average(ctr2_performance, axis=0)
    avg_ctr3_performance = np.average(ctr3_performance, axis=0)
    return avg_exp1_performance, avg_exp2_performance, avg_ctr1_performance, avg_ctr2_performance, avg_ctr3_performance


def exp_diff_merge_interval(env_class: Type[MyEnv],
                            agent_trainer,
                            log_dir,
                            dim_embedding,
                            act_total_train_embeddings,
                            act_total_validation_embeddings,
                            crt_total_train_embeddings,
                            crt_total_validation_embeddings,
                            recorder_extra_hyperparameters: dict,
                            merge_interval_set: list,
                            gamma=GAMMA,
                            total_frame_upper_bound=int(1e+6),
                            ):

    merge_interval_set_size = len(merge_interval_set)

    # Initialize train, validation, test theta set

    train_theta_set = np.load(log_dir + 'train_theta.npy')
    validation_theta_set = np.load(log_dir + 'validation_theta.npy')

    for count in tqdm(range(outer_loop_num)):
        # Train our algorithm with different merge interval
        for k in range(merge_interval_set_size):
            merge_interval = merge_interval_set[k]
            current_log_dir = log_dir + 'm=' + str(merge_interval) + 'out=' + str(count) + '/'
            # Call DDPG Avg
            act_net, crt_net, act_embeddings, crt_embeddings, _, _ =\
                Per_DDPGAvg(env_class=env_class,
                            recorder_extra_hyperparameters=recorder_extra_hyperparameters,
                            dim_embedding=dim_embedding,
                            act_train_embeddings=act_total_train_embeddings[count],
                            act_validation_embeddings=act_total_validation_embeddings[count],
                            crt_train_embeddings=crt_total_train_embeddings[count],
                            train_theta_set=train_theta_set[count],
                            validation_theta_set=validation_theta_set,
                            agent_trainer=agent_trainer,
                            gamma=gamma,
                            merge_interval=merge_interval,
                            total_frame_upper_bound=total_frame_upper_bound,
                            log_dir=current_log_dir,
                            crt_flag=True,
                            silent_flag=False,
                            threshold=threshold)
            act_net_para = act_net.state_dict()
            crt_net_para = crt_net.state_dict()
            torch.save(act_net_para, current_log_dir+'act_net_para.pt')
            torch.save(crt_net_para, current_log_dir+'crt_net_para.pt')

            # M-net for comparison
            M_act_net, M_crt_net, _, _ = M_DDPGAvg(env_class=Ori_MyHalfCheetahEnv,
                                                   recorder_extra_hyperparameters=recorder_extra_hyperparameters,
                                                   train_theta_set=train_theta_set[count],
                                                   validation_theta_set=validation_theta_set,
                                                   agent_trainer=M_DDPG_TRAIN,
                                                   gamma=gamma,
                                                   merge_interval=merge_interval,
                                                   total_frame_upper_bound=total_frame_upper_bound,
                                                   log_dir=current_log_dir,
                                                   crt_flag=True,
                                                   silent_flag=False,
                                                   threshold=threshold)
            M_act_net_para = M_act_net.state_dict()
            M_crt_net_para = M_crt_net.state_dict()
            torch.save(M_act_net_para, current_log_dir+'M_act_net_para.pt')
            torch.save(M_crt_net_para, current_log_dir+'M_crt_net_para.pt')

            # Personalization
            avg_exp1_performance, avg_exp2_performance, avg_ctr1_performance\
                , avg_ctr2_performance, avg_ctr3_performance = \
                integrate_personalization(
                    Act_net=act_net,
                    Crt_net=crt_net,
                    M_act_net=M_act_net,
                    M_crt_net=M_crt_net,
                    dim_embedding=dim_embedding,
                    act_validation_embeddings=act_total_validation_embeddings[count],
                    crt_validation_embeddings=crt_total_validation_embeddings[count],
                    validation_theta_set=validation_theta_set,
                    gamma=gamma,
                    log_dir=current_log_dir,
                    threshold=threshold)

    return


def report_diff_merge_interval(
                               merge_interval_set: list,
                               total_frame_upper_bound):

    log_dir = global_log_dir
    if not os.path.exists('./' + save_dir_name):
        os.makedirs('./' + save_dir_name)

    min_merge_interval = min(merge_interval_set)
    report_size = total_frame_upper_bound // min_merge_interval  # We assume this is an exact division

    total_cum_reward_E1 = np.zeros((outer_loop_num, report_size))
    total_cum_reward_E2 = np.zeros((outer_loop_num, report_size))

    total_test_cum_reward_E1 = np.zeros((outer_loop_num, report_size))
    total_test_cum_reward_E2 = np.zeros((outer_loop_num, report_size))

    M_total_cum_reward_E1 = np.zeros((outer_loop_num, report_size))
    M_total_cum_reward_E2 = np.zeros((outer_loop_num, report_size))

    M_total_test_cum_reward_E1 = np.zeros((outer_loop_num, report_size))
    M_total_test_cum_reward_E2 = np.zeros((outer_loop_num, report_size))

    for count in tqdm(range(outer_loop_num)):
        cum_reward_E1 = np.load(log_dir + f'm={merge_interval_set[0]}out={count}/train/cum_reward_avg.npy')
        test_cum_reward_E1 = np.load(log_dir + f'm={merge_interval_set[0]}out={count}/validation/cum_reward_avg.npy')
        tmp_cum_reward_E2 = np.load(log_dir + f'm={merge_interval_set[1]}out={count}/train/cum_reward_avg.npy')
        tmp_test_cum_reward_E2 = np.load(log_dir + f'm={merge_interval_set[1]}out={count}/validation/cum_reward_avg.npy')

        M_cum_reward_E1 = np.load(log_dir + f'm={merge_interval_set[0]}out={count}/train_M/cum_reward_avg.npy')
        M_test_cum_reward_E1 = np.load(log_dir + f'm={merge_interval_set[0]}out={count}/validation_M/cum_reward_avg.npy')
        M_tmp_cum_reward_E2 = np.load(log_dir + f'm={merge_interval_set[1]}out={count}/train_M/cum_reward_avg.npy')
        M_tmp_test_cum_reward_E2 = np.load(log_dir + f'm={merge_interval_set[1]}out={count}/validation_M/cum_reward_avg.npy')

        E12ratio = merge_interval_set[1] // merge_interval_set[0]
        cum_reward_E2 = np.zeros((report_size))
        test_cum_reward_E2 = np.zeros((report_size))

        M_cum_reward_E2 = np.zeros((report_size))
        M_test_cum_reward_E2 = np.zeros((report_size))
        for i in range(report_size // E12ratio):
            cum_reward_E2[i*E12ratio:(i+1)*E12ratio] = tmp_cum_reward_E2[i]
            test_cum_reward_E2[i*E12ratio:(i+1)*E12ratio] = tmp_test_cum_reward_E2[i]
            M_cum_reward_E2[i*E12ratio:(i+1)*E12ratio] = M_tmp_cum_reward_E2[i]
            M_test_cum_reward_E2[i*E12ratio:(i+1)*E12ratio] = M_tmp_test_cum_reward_E2[i]

        total_cum_reward_E1[count] = cum_reward_E1
        total_cum_reward_E2[count] = cum_reward_E2
        total_test_cum_reward_E1[count] = test_cum_reward_E1
        total_test_cum_reward_E2[count] = test_cum_reward_E2

        M_total_cum_reward_E1[count] = M_cum_reward_E1
        M_total_cum_reward_E2[count] = M_cum_reward_E2
        M_total_test_cum_reward_E1[count] = M_test_cum_reward_E1
        M_total_test_cum_reward_E2[count] = M_test_cum_reward_E2

    np.save('./' + save_dir_name + '/MEOBj_lst.npy', total_cum_reward_E1)
    np.save('./' + save_dir_name + '/MEOBj3_lst.npy', total_cum_reward_E2)

    np.save('./' + save_dir_name + '/test_MEOBj_lst.npy', total_test_cum_reward_E1)
    np.save('./' + save_dir_name + '/test_MEOBj3_lst.npy', total_test_cum_reward_E2)

    np.save('./' + save_dir_name + '/M_MEOBj_lst.npy', M_total_cum_reward_E1)
    np.save('./' + save_dir_name + '/M_MEOBj3_lst.npy', M_total_cum_reward_E2)

    np.save('./' + save_dir_name + '/M_test_MEOBj_lst.npy', M_total_test_cum_reward_E1)
    np.save('./' + save_dir_name + '/M_test_MEOBj3_lst.npy', M_total_test_cum_reward_E2)
    return


if __name__ == '__main__':
    integrate_diff_merge_interval()
