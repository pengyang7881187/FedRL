import os
import copy
import math
import ptan
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib as mpl
mpl.use('Agg')
from Config import *
from DQNAvg import net_para_add, net_para_scale
from DeepRLAlgo import RLNet, double_DQN, MLP_Q_Net, unpack_batch
from Per_ExpRecorder import AbsoluteRewardRecorder as Per_AbsoluteRewardRecorder
from ExpRecorder import AbsoluteRewardRecorder as Ori_AbsoluteRewardRecorder
from Per_MyAcrobot import MyAcrobotEnv as Per_MyAcrobotEnv
from MyAcrobot import Mass1_MyAcrobotEnv as Ori_MyAcrobotEnv
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--lr_embedding", default=1e-4, type=float)
parser.add_argument("--train_size", default=10, type=int)
parser.add_argument("--validation_size", default=10, type=int)
parser.add_argument("--embed_type", default="Gauss", type=str)
parser.add_argument("--outer_loop_num", default=100, type=int)
parser.add_argument("--continue_ub", default=50, type=int)
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

os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)

save_dir_name = 'plot-PerDQNAvg-Acrobot'
global_log_dir = './PerDQNAvg-Acrobot/'


class Per_MLP_Q_Net(RLNet):
    def __init__(self, obs_size, n_actions, hidden_size, dim_embedding):
        super(Per_MLP_Q_Net, self).__init__(obs_size, n_actions)
        self.network_type = 'Q'
        self.dim_embedding = dim_embedding
        self.net = nn.Sequential(
            nn.Linear(self.obs_size + dim_embedding, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x_em):
        return self.net(x_em.float())


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


@torch.no_grad()
def Per_unpack_batch(batch, net, gamma, update_embedding=False, embedding=None):
    # embedding should be numpy array
    states = []
    actions = []
    rewards = []
    done_masks = []
    last_states = []
    for exp in batch:
        state = exp.state
        if update_embedding:
            dim_embedding = embedding.size
            state[:dim_embedding] = embedding
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        done_masks.append(exp.last_state is None)
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

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    last_states_v = torch.tensor(last_states).to(device)
    last_state_q_v = net(last_states_v)
    best_last_q_v = torch.max(last_state_q_v, dim=1)[0]
    best_last_q_v[done_masks] = 0.0
    return states_v, actions_v, best_last_q_v * gamma + rewards_v


# Use epsilon-greedy to explore
def Per_double_DQN(
               env: Per_MyAcrobotEnv,
               embedding,
               hidden_size,
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

    dim_embedding = embedding.size
    env.embedding = embedding

    embedding_tensor = (torch.tensor(embedding)).to(device)
    embedding_tensor.requires_grad_(True)

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    if continue_train_flag:
        net = Per_MLP_Q_Net(obs_size=obs_size, n_actions=n_actions, hidden_size=hidden_size,
                            dim_embedding=dim_embedding)
        net.load_state_dict(continue_train_para)
        net = net.to(device)
    else:
        net = Per_MLP_Q_Net(obs_size=obs_size, n_actions=n_actions, hidden_size=hidden_size,
                            dim_embedding=dim_embedding).to(device)
    tgt_net = ptan.agent.TargetNet(net)

    def state_preprocessor(states):
        embedding_tensor_clone = embedding_tensor.clone()
        if torch.is_tensor(states):
            t_states = (states.clone().detach()).to(device)
        else:
            t_states = (torch.tensor(states)).to(device)
        if t_states.ndim == 1 or t_states.shape[1] == 1:
            t_states[:dim_embedding] = embedding_tensor_clone
        else:
            t_states[:, :dim_embedding] = embedding_tensor_clone
        return t_states

    def state_preprocessor_for_tensor(t_states):
        if t_states.ndim == 1 or t_states.shape[1] == 1:
            t_states[:dim_embedding] = embedding_tensor
        else:
            t_states[:, :dim_embedding] = embedding_tensor
        return t_states

    selector = ptan.actions.ArgmaxActionSelector()
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=epsilon, selector=selector)
    agent = ptan.agent.DQNAgent(net, selector, device=device, preprocessor=state_preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=gamma)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=replay_size)

    parameters = [
        {'params': net.parameters()},
        {'params': [embedding_tensor], 'lr': LR_EMBED}
    ]
    optimizer = optim.Adam(parameters, lr)

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
        states_v, actions_v, tgt_q_v = Per_unpack_batch(batch, tgt_net.target_model, gamma, True, embedding)
        optimizer.zero_grad()
        states_v = state_preprocessor_for_tensor(states_v)
        q_v = net(states_v)
        q_v = q_v.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        loss_v = F.mse_loss(q_v, tgt_q_v)
        loss_v.backward()
        optimizer.step()
        selector.epsilon *= epsilon_decay
        embedding = embedding_tensor.clone().cpu().detach().numpy()

        if step % sync_interval == 0:
            tgt_net.sync()
    # You need to reconstruct buffer using buffer.buffer
    return net, embedding, buffer.buffer


def Per_QAvg(
         dim_embedding,
         train_embeddings,
         validation_embeddings,
         hidden_size,
         recorder_extra_hyperparameters: dict,  # Do not parse None to it!
         train_theta_set: np.ndarray,
         validation_theta_set: np.ndarray,
         agent_trainer_hyperparameters: dict,  # Do not parse None to it!
         agent_trainer=Per_double_DQN,
         gamma=GAMMA,
         merge_interval=MERGE_INTERVAL,
         show_interval=SHOW_INTERVAL,
         train_episodes_upper_bound=TRAIN_EPISODES_UB,
         continue_train_flag=False,
         continue_train_para=None,
         exploration_strategy='EPSILON_GREEDY',
         log_dir='./log/',
         silent_flag=True):
    env_class = Per_MyAcrobotEnv
    recorder_class = Per_AbsoluteRewardRecorder

    # mkdir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    os.makedirs(log_dir + 'train/')
    os.makedirs(log_dir + 'validation/')

    assert train_episodes_upper_bound % merge_interval == 0  # This assumption is for convenience

    # Prepare environments and set some parameters
    # We do not need validation_size and test_size in this function
    train_size = train_theta_set.size
    env_train = []
    for theta in train_theta_set:
        env_train.append(env_class(para=theta))
    obs_size = env_train[0].observation_space.shape[0]
    n_actions = env_train[0].action_space.n

    # Modify agent_extra_hyperparameters
    agent_trainer_hyperparameters['train_episodes_upper_bound'] = merge_interval

    # Prepare recorder
    # Outer loop: train agents and merge their parameters
    outer_loop_iter_upper_bound = int(train_episodes_upper_bound / merge_interval)
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
                                    embedding_set=train_embeddings,
                                    **recorder_extra_hyperparameters)
    validation_recorder = recorder_class(env_class=env_class,
                                         theta_set=validation_theta_set,
                                         merge_interval=merge_interval,
                                         log_folder_dir=validation_recorder_log_dir,
                                         record_len=record_len,
                                         recorder_name=validation_recorder_name,
                                         gamma=gamma,
                                         embedding_set=validation_embeddings,
                                         **recorder_extra_hyperparameters)

    # Initialize network or load network parameter
    net = Per_MLP_Q_Net(obs_size=obs_size, n_actions=n_actions, hidden_size=hidden_size,
                        dim_embedding=dim_embedding)
    if continue_train_flag is True:
        if continue_train_para is not None:
            net.load_state_dict(continue_train_para)
        else:
            # Correct the flag
            continue_train_flag = False
    net = net.to(device)  # Load the net to GPU

    # For epsilon-greedy exploration strategy, we need to decay epsilon during training:
    current_init_epsilon = 0  # This is only for avoiding warning
    epsilon_decay = 0  # This is only for avoiding warning
    if exploration_strategy == 'EPSILON_GREEDY':
        current_init_epsilon = agent_trainer_hyperparameters['epsilon']
        epsilon_decay = agent_trainer_hyperparameters['epsilon_decay']

    embeddings = train_embeddings.copy()

    # Outer loop: train agents and merge their parameters
    for i in range(outer_loop_iter_upper_bound):
        #########
        # Train #
        #########
        net_para = 0  # This is only for avoiding warning
        # Decay epsilon for epsilon-greedy
        if exploration_strategy == 'EPSILON_GREEDY':
            agent_trainer_hyperparameters['epsilon'] = current_init_epsilon
            current_init_epsilon *= (epsilon_decay ** merge_interval)
        # First round of iteration when continue flag is False
        if i == 0 and continue_train_flag is False:
            for j in range(train_size):
                tmp_net, tmp_embedding, _ = agent_trainer(env=env_train[j],
                                                          embedding=embeddings[j],
                                                          hidden_size=hidden_size,
                                                          **agent_trainer_hyperparameters)
                tmp_para = tmp_net.state_dict()
                embeddings[j] = tmp_embedding.copy()
                if j == 0:
                    net_para = copy.deepcopy(tmp_para)
                else:
                    net_para = net_para_add(net_para, tmp_para)
        # Other rounds of iteration, continue train from net
        else:
            for j in range(train_size):
                tmp_net, tmp_embedding, _ = agent_trainer(env=env_train[j],
                                                          embedding=embeddings[j],
                                                          hidden_size=hidden_size,
                                                          continue_train_flag=True,
                                                          continue_train_para=net.state_dict(),
                                                          **agent_trainer_hyperparameters
                                                          )
                tmp_para = tmp_net.state_dict()
                embeddings[j] = tmp_embedding.copy()
                if j == 0:
                    net_para = copy.deepcopy(tmp_para)
                else:
                    net_para = net_para_add(net_para, tmp_para)
        net_para = net_para_scale(net_para, (1 / train_size))
        net.load_state_dict(net_para)

        ################
        # Record error #
        ################

        train_recorder.embedding_set = embeddings.copy()

        # Record training error
        train_recorder.record(net=net)
        # Record validation error
        validation_recorder.record(net=net)

        if not silent_flag:
            # Report error
            if (i+1) % show_interval == 0:
                train_recorder.report()
                validation_recorder.report()
    # End outer loop here
    return net, embeddings, train_recorder, validation_recorder


def M_QAvg(
         hidden_size,
         recorder_extra_hyperparameters: dict,  # Do not parse None to it!
         train_theta_set: np.ndarray,
         validation_theta_set: np.ndarray,
         agent_trainer_hyperparameters: dict,  # Do not parse None to it!
         gamma=GAMMA,
         merge_interval=MERGE_INTERVAL,
         show_interval=SHOW_INTERVAL,
         train_episodes_upper_bound=TRAIN_EPISODES_UB,
         continue_train_flag=False,
         continue_train_para=None,
         exploration_strategy='EPSILON_GREEDY',
         log_dir='./log/',
         silent_flag=True):
    env_class = Ori_MyAcrobotEnv
    recorder_class = Ori_AbsoluteRewardRecorder
    agent_trainer = double_DQN

    # mkdir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    os.makedirs(log_dir + 'train_M/')
    os.makedirs(log_dir + 'validation_M/')

    assert train_episodes_upper_bound % merge_interval == 0  # This assumption is for convenience

    # Prepare environments and set some parameters
    # We do not need validation_size and test_size in this function
    train_size = train_theta_set.size
    env_train = []
    for theta in train_theta_set:
        env_train.append(env_class(para=theta))
    obs_size = env_train[0].observation_space.shape[0]
    n_actions = env_train[0].action_space.n

    # Modify agent_extra_hyperparameters
    agent_trainer_hyperparameters['train_episodes_upper_bound'] = merge_interval

    net_extra_hyperparameters = {'hidden_size': hidden_size}

    # Prepare recorder
    # Outer loop: train agents and merge their parameters
    outer_loop_iter_upper_bound = int(train_episodes_upper_bound / merge_interval)
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
    net = MLP_Q_Net(obs_size=obs_size, n_actions=n_actions, hidden_size=hidden_size)
    if continue_train_flag is True:
        if continue_train_para is not None:
            net.load_state_dict(continue_train_para)
        else:
            # Correct the flag
            continue_train_flag = False
    net = net.to(device)  # Load the net to GPU

    # For epsilon-greedy exploration strategy, we need to decay epsilon during training:
    current_init_epsilon = 0  # This is only for avoiding warning
    epsilon_decay = 0  # This is only for avoiding warning
    if exploration_strategy == 'EPSILON_GREEDY':
        current_init_epsilon = agent_trainer_hyperparameters['epsilon']
        epsilon_decay = agent_trainer_hyperparameters['epsilon_decay']

    # Outer loop: train agents and merge their parameters
    for i in range(outer_loop_iter_upper_bound):
        #########
        # Train #
        #########
        net_para = 0  # This is only for avoiding warning
        # Decay epsilon for epsilon-greedy
        if exploration_strategy == 'EPSILON_GREEDY':
            agent_trainer_hyperparameters['epsilon'] = current_init_epsilon
            current_init_epsilon *= (epsilon_decay ** merge_interval)
        # First round of iteration when continue flag is False
        if i == 0 and continue_train_flag is False:
            for j in range(train_size):
                tmp_net, _ = agent_trainer(env=env_train[j],
                                           net_extra_hyperparameters=net_extra_hyperparameters,
                                           **agent_trainer_hyperparameters)
                tmp_para = tmp_net.state_dict()
                if j == 0:
                    net_para = copy.deepcopy(tmp_para)
                else:
                    net_para = net_para_add(net_para, tmp_para)
        # Other rounds of iteration, continue train from net
        else:
            for j in range(train_size):
                tmp_net, _ = agent_trainer(env=env_train[j],
                                           net_extra_hyperparameters=net_extra_hyperparameters,
                                           continue_train_flag=True,
                                           continue_train_para=net.state_dict(),
                                           **agent_trainer_hyperparameters
                                           )
                tmp_para = tmp_net.state_dict()
                if j == 0:
                    net_para = copy.deepcopy(tmp_para)
                else:
                    net_para = net_para_add(net_para, tmp_para)
        net_para = net_para_scale(net_para, (1 / train_size))
        net.load_state_dict(net_para)

        ################
        # Record error #
        ################

        # Record training error
        train_recorder.record(net=net)
        # Record validation error
        validation_recorder.record(net=net)

        if not silent_flag:
            # Report error
            if (i+1) % show_interval == 0:
                train_recorder.report()
                validation_recorder.report()
    # End outer loop here
    return net, train_recorder, validation_recorder


@torch.no_grad()
def Per_evaluate_expected_reward(net, env, evaluate_episodes_for_eval=10, gamma=GAMMA, replay_size=REPLAY_SIZE):
    # For Q-network only
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


def integrate_personalization(
                     net: Per_MLP_Q_Net,
                     M_net: MLP_Q_Net,
                     dim_embedding,
                     validation_embeddings,
                     hidden_size,
                     validation_theta_set,
                     gamma,
                     log_dir,
                     silent_flag=True,
                     init_embedding_flag=False,
                     init_embeddings=None):

    env_class = Per_MyAcrobotEnv
    ori_env_class = Ori_MyAcrobotEnv

    current_log_dir = log_dir + 'person/'
    if not os.path.exists(current_log_dir):
        os.makedirs(current_log_dir)

    validation_size = validation_theta_set.size

    validation_embeddings_for_use = validation_embeddings.copy()

    if init_embedding_flag:
        avg_embeddings = np.average(init_embeddings, axis=0)
        for ttt in range(validation_size):
            validation_embeddings_for_use[ttt] = avg_embeddings.copy()

    env_validation = []
    for theta in validation_theta_set:
        env_validation.append(env_class(para=theta))
    obs_size = env_validation[0].observation_space.shape[0]
    n_actions = env_validation[0].action_space.n

    def exp_personalization(ori_net, type):
        ori_para = ori_net.state_dict()

        performance = np.zeros((validation_size, CONTINUE_TRAIN_UB))

        current_validation_embeddings = validation_embeddings_for_use.copy()

        for i in range(validation_size):
            env = env_class(para=validation_theta_set[i])
            embedding = copy.deepcopy(current_validation_embeddings[i])
            env.embedding = embedding
            embedding_tensor = (torch.tensor(embedding)).to(device)
            embedding_tensor.requires_grad_(True)

            net_exp_para = copy.deepcopy(ori_para)
            net_exp = Per_MLP_Q_Net(obs_size=obs_size, n_actions=n_actions, hidden_size=hidden_size,
                                dim_embedding=dim_embedding)
            net_exp.load_state_dict(net_exp_para)
            net_exp = net_exp.to(device)

            epsilon = 0.1
            lr = LR_PER
            lr_embedding = LR_EMBED
            replay_size = 1000
            batch_size = 4
            epsilon_decay = 0.99
            sync_interval = 10

            tgt_net = ptan.agent.TargetNet(net_exp)

            def state_preprocessor(states):
                if torch.is_tensor(states):
                    t_states = (states.clone().detach()).to(device)
                else:
                    t_states = (torch.tensor(states)).to(device)
                if t_states.ndim == 1 or t_states.shape[1] == 1:
                    t_states[:dim_embedding] = embedding_tensor
                else:
                    t_states[:, :dim_embedding] = embedding_tensor
                return t_states

            def state_preprocessor_for_tensor(t_states):
                if t_states.ndim == 1 or t_states.shape[1] == 1:
                    t_states[:dim_embedding] = embedding_tensor
                else:
                    t_states[:, :dim_embedding] = embedding_tensor
                return t_states

            selector = ptan.actions.ArgmaxActionSelector()
            selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=epsilon, selector=selector)
            agent = ptan.agent.DQNAgent(net_exp, selector, device=device, preprocessor=state_preprocessor)
            exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=gamma)
            buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=replay_size)

            if type == 'Exp-1':
                # Embedding only
                optimizer = optim.Adam([embedding_tensor], lr_embedding)
            elif type == 'Exp-2':
                # Both embedding and net
                parameters = [
                    {'params': net_exp.parameters()},
                    {'params': [embedding_tensor], 'lr': lr_embedding}
                    ]
                optimizer = optim.Adam(parameters, lr)
            elif type == 'Ctr-1':
                # Net only
                optimizer = optim.Adam(net_exp.parameters(), lr)
            elif type == 'Ctr-2':
                # Not update
                optimizer = None
            else:
                raise

            step = 0
            episode = 0
            evaluate_reward = 0  # Used for evaluate model's performance

            while True:
                step += 1
                buffer.populate(1)

                for reward, steps in exp_source.pop_rewards_steps():
                    # evaluate here
                    tmp_env = env_class(para=validation_theta_set[i])
                    tmp_env.embedding = embedding

                    performance[i][episode] = Per_evaluate_expected_reward(net=net_exp, env=tmp_env,
                                                                           evaluate_episodes_for_eval=30,
                                                                           gamma=gamma, replay_size=replay_size)
                    episode += 1
                    if episode >= CONTINUE_TRAIN_UB:
                        break
                    evaluate_reward += reward
                    if not silent_flag:
                        print("%d: episode %d done, reward=%.3f, epsilon=%.2f" % (
                            step, episode, reward, selector.epsilon))
                if episode >= CONTINUE_TRAIN_UB:
                    if not silent_flag:
                        print("Exceed episode upper bound!")
                    break

                if len(buffer) < 2 * batch_size:
                    continue

                batch = buffer.sample(batch_size)
                states_v, actions_v, tgt_q_v = Per_unpack_batch(batch, tgt_net.target_model, gamma, True, embedding)
                states_v = state_preprocessor_for_tensor(states_v)
                if type != 'Ctr-2':
                    optimizer.zero_grad()
                q_v = net_exp(states_v)
                q_v = q_v.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
                loss_v = F.mse_loss(q_v, tgt_q_v)
                if type != 'Ctr-2':
                    loss_v.backward()
                    optimizer.step()
                embedding = embedding_tensor.clone().cpu().detach().numpy()
                selector.epsilon *= epsilon_decay

                if step % sync_interval == 0:
                    tgt_net.sync()
        return performance

    def ori_personalization(ori_net):
        ori_para = ori_net.state_dict()

        performance = np.zeros((validation_size, CONTINUE_TRAIN_UB))

        for i in range(validation_size):
            env = ori_env_class(para=validation_theta_set[i])
            net_exp_para = copy.deepcopy(ori_para)
            net_exp = MLP_Q_Net(obs_size=obs_size, n_actions=n_actions, hidden_size=hidden_size)
            net_exp.load_state_dict(net_exp_para)
            net_exp = net_exp.to(device)

            epsilon = 0.1
            lr = LR_PER
            replay_size = 1000
            batch_size = 4
            epsilon_decay = 0.99
            sync_interval = 10

            tgt_net = ptan.agent.TargetNet(net_exp)

            selector = ptan.actions.ArgmaxActionSelector()
            selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=epsilon, selector=selector)
            agent = ptan.agent.DQNAgent(net_exp, selector, device=device)
            exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=gamma)
            buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=replay_size)

            optimizer = optim.Adam(net_exp.parameters(), lr)

            step = 0
            episode = 0
            evaluate_reward = 0  # Used for evaluate model's performance

            while True:
                step += 1
                buffer.populate(1)

                for reward, steps in exp_source.pop_rewards_steps():
                    # evaluate here
                    tmp_env = ori_env_class(para=validation_theta_set[i])

                    performance[i][episode] = net_exp.evaluate_expected_reward(env=tmp_env,
                                                                               evaluate_episodes_for_eval=30,
                                                                               gamma=gamma, replay_size=replay_size)
                    episode += 1
                    if episode >= CONTINUE_TRAIN_UB:
                        break
                    evaluate_reward += reward
                    if not silent_flag:
                        print("%d: episode %d done, reward=%.3f, epsilon=%.2f" % (
                            step, episode, reward, selector.epsilon))
                if episode >= CONTINUE_TRAIN_UB:
                    if not silent_flag:
                        print("Exceed episode upper bound!")
                    break

                if len(buffer) < 2 * batch_size:
                    continue

                batch = buffer.sample(batch_size)
                states_v, actions_v, tgt_q_v = unpack_batch(batch, tgt_net.target_model, gamma)
                optimizer.zero_grad()
                q_v = net_exp(states_v)
                q_v = q_v.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
                loss_v = F.mse_loss(q_v, tgt_q_v)
                loss_v.backward()
                optimizer.step()
                selector.epsilon *= epsilon_decay
                if step % sync_interval == 0:
                    tgt_net.sync()
        return performance

    # Numpy array of shape (validation_size, CONTINUE_TRAIN_UB)
    exp1_performance = exp_personalization(net, 'Exp-1')
    exp2_performance = exp_personalization(net, 'Exp-2')
    ctr1_performance = exp_personalization(net, 'Ctr-1')
    ctr2_performance = exp_personalization(net, 'Ctr-2')
    ctr3_performance = ori_personalization(M_net)

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


# We separate train and report, cuz train is very time-consuming, while report is not
def integrate_diff_merge_interval():
    log_dir = global_log_dir
    outer_loop_num = OUTER_LOOP_NUM
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists('./' + save_dir_name):
        os.makedirs('./' + save_dir_name)

    np.random.seed(seed)

    # Prepare some parameters: most are loaded from Config.py

    gamma = GAMMA
    lr = LR
    train_episodes_upper_bound = 20

    train_size = TRAIN_SIZE_PER
    validation_size = VALIDATION_SIZE_PER

    embed_mode = EMBED_TYPE

    train_theta_set = np.random.uniform(size=(outer_loop_num, train_size))
    validation_theta_set = np.linspace(0, 1, validation_size)
    np.save(log_dir + 'train_theta.npy', train_theta_set)
    np.save(log_dir + 'validation_theta.npy', validation_theta_set)

    dim_embedding, _, _ = \
        init_embedding(train_size, validation_size, mode=embed_mode)

    total_train_embeddings = np.zeros((outer_loop_num, train_size, dim_embedding))
    total_validation_embeddings = np.zeros((outer_loop_num, validation_size, dim_embedding))
    for i in range(outer_loop_num):
        _, train_embeddings, validation_embeddings = \
            init_embedding(train_size, validation_size, mode=embed_mode)
        total_train_embeddings[i] = train_embeddings.copy()
        total_validation_embeddings[i] = validation_embeddings.copy()
    np.save(log_dir + 'total_train_embedding.npy', total_train_embeddings)
    np.save(log_dir + 'total_validation_embedding.npy', total_validation_embeddings)

    hidden_size = 128

    init_epsilon = INIT_EPSILON  # While training in QAvg, init_epsilon will also decay
    epsilon_decay = EPS_DECAY  # This is epsilon decay in double DQN, not in QAvg
    tgt_net_sync = TGT_NET_SYNC
    batch_size = BATCH_SIZE
    evaluate_episodes_for_conv = EVALUATE_EPISODES_FOR_CONV
    solve_criterion = SOLVE_CRITERION
    replay_size = REPLAY_SIZE
    agent_trainer = Per_double_DQN
    # train_episodes_upper_bound should be specified later on
    agent_trainer_hyperparameters = {'epsilon': init_epsilon,
                                     'epsilon_decay': epsilon_decay,
                                     'sync_interval': tgt_net_sync,
                                     'gamma': gamma,
                                     'batch_size': batch_size,
                                     'lr': lr,
                                     'evaluate_episodes_for_conv': evaluate_episodes_for_conv,
                                     'solve_criterion': solve_criterion,
                                     'replay_size': replay_size,
                                     'silent_flag': True}
    show_interval = SHOW_INTERVAL
    evaluate_episodes_for_eval = EVALUATE_EPISODES_FOR_EVAL
    recorder_extra_hyperparameters = {'evaluate_episodes_for_eval': evaluate_episodes_for_eval}

    merge_interval_set = [1, 2]

    exp_diff_merge_interval(agent_trainer=agent_trainer,
                            hidden_size=hidden_size,
                            dim_embedding=dim_embedding,
                            total_train_embeddings=total_train_embeddings,
                            total_validation_embeddings=total_validation_embeddings,
                            recorder_extra_hyperparameters=recorder_extra_hyperparameters,
                            agent_trainer_hyperparameters=agent_trainer_hyperparameters,
                            merge_interval_set=merge_interval_set,
                            gamma=gamma,
                            train_episodes_upper_bound=train_episodes_upper_bound,
                            show_interval=show_interval,
                            log_dir=log_dir,
                            outer_loop_num=outer_loop_num)
    report_diff_merge_interval(
                               outer_loop_num=outer_loop_num,
                               train_episodes_upper_bound=train_episodes_upper_bound,
                               merge_interval_set=merge_interval_set)
    return


def exp_diff_merge_interval(agent_trainer,
                            hidden_size,
                            dim_embedding,
                            total_train_embeddings,
                            total_validation_embeddings,
                            recorder_extra_hyperparameters: dict,
                            agent_trainer_hyperparameters: dict,
                            merge_interval_set: list,
                            gamma=GAMMA,
                            train_episodes_upper_bound=TRAIN_EPISODES_UB,
                            show_interval=SHOW_INTERVAL,
                            log_dir=None,
                            outer_loop_num=10):
    merge_interval_set_size = len(merge_interval_set)

    # Initialize train, validation, test theta set

    train_theta_set = np.load(log_dir + 'train_theta.npy')
    validation_theta_set = np.load(log_dir + 'validation_theta.npy')

    for count in tqdm(range(outer_loop_num)):
        # Train our algorithm with different merge interval
        for k in range(merge_interval_set_size):
            merge_interval = merge_interval_set[k]
            # print('Train QAvg with merge interval=' + str(merge_interval) + ':')
            current_log_dir = log_dir + 'm=' + str(merge_interval) + 'out=' + str(count) + '/'
            # Call QAvg
            net, embeddings, _, _ =\
                Per_QAvg(
                     dim_embedding=dim_embedding,
                     train_embeddings=total_train_embeddings[count],
                     validation_embeddings=total_validation_embeddings[count],
                     hidden_size=hidden_size,
                     recorder_extra_hyperparameters=recorder_extra_hyperparameters,
                     train_theta_set=train_theta_set[count],
                     validation_theta_set=validation_theta_set,
                     agent_trainer=agent_trainer,
                     agent_trainer_hyperparameters=agent_trainer_hyperparameters,
                     gamma=gamma,
                     merge_interval=merge_interval,
                     show_interval=show_interval,
                     train_episodes_upper_bound=train_episodes_upper_bound,
                     log_dir=current_log_dir,
                     silent_flag=True)
            net_para = net.state_dict()
            torch.save(net_para, current_log_dir+'net_para.pt')

            # M-net for comparison
            M_net, _, _ =\
                M_QAvg(
                     hidden_size=hidden_size,
                     recorder_extra_hyperparameters=recorder_extra_hyperparameters,
                     train_theta_set=train_theta_set[count],
                     validation_theta_set=validation_theta_set,
                     agent_trainer_hyperparameters=agent_trainer_hyperparameters,
                     gamma=gamma,
                     merge_interval=merge_interval,
                     show_interval=show_interval,
                     train_episodes_upper_bound=train_episodes_upper_bound,
                     log_dir=current_log_dir,
                     silent_flag=True)
            M_net_para = M_net.state_dict()
            torch.save(M_net_para, current_log_dir+'M_net_para.pt')

            # Personalization
            avg_exp1_performance, avg_exp2_performance, avg_ctr1_performance\
                , avg_ctr2_performance, avg_ctr3_performance = \
                integrate_personalization(
                    net=net,
                    M_net=M_net,
                    dim_embedding=dim_embedding,
                    validation_embeddings=total_validation_embeddings[count],
                    hidden_size=hidden_size,
                    validation_theta_set=validation_theta_set,
                    gamma=gamma,
                    log_dir=current_log_dir,
                    silent_flag=True)
    return


# We could write report function for recorder class, i.e. write this function by calling recorder class method
def report_diff_merge_interval(
                               outer_loop_num,
                               merge_interval_set: list,
                               train_episodes_upper_bound=TRAIN_EPISODES_UB):
    log_dir = global_log_dir

    min_merge_interval = min(merge_interval_set)
    report_size = train_episodes_upper_bound // min_merge_interval  # We assume this is an exact division

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
