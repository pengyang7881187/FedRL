import copy
from DQNAvg import DQNAvg, net_para_add, net_para_scale
from DeepRLAlgo import *
from ExpRecorder import ExperimentRecorder, AbsoluteRewardRecorder
from typing import Type
from MyAcrobot import MyAcrobotEnv
from tqdm import tqdm

outer_loop_num = 10

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

log_dir = './DQNAvg-Acrobot/'
save_dir = 'plot-DQNAvg-Acrobot'


# We separate train and report, cuz train is very time-consuming, while report is not
def integrate_diff_merge_interval():
    np.random.seed(251)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # Prepare some parameters: most are loaded from Config.py
    env_class = MyAcrobotEnv
    net_class = MLP_Q_Net
    gamma = GAMMA
    lr = 1e-4
    train_episodes_upper_bound = 64

    train_size = 10
    validation_size = 10
    test_size = 2

    train_theta_set = np.random.uniform(size=(outer_loop_num, train_size))
    validation_theta_set = np.linspace(0, 1, validation_size)
    test_theta_set = np.random.uniform(size=test_size)
    np.save(log_dir + 'train_theta.npy', train_theta_set)
    np.save(log_dir + 'validation_theta.npy', validation_theta_set)
    np.save(log_dir + 'test_theta.npy', test_theta_set)

    init_epsilon = INIT_EPSILON  # While training in QAvg, init_epsilon will also decay
    epsilon_decay = EPS_DECAY  # This is epsilon decay in double DQN, not in QAvg
    tgt_net_sync = TGT_NET_SYNC
    batch_size = BATCH_SIZE
    hidden_size = HIDDEN_SIZE  # Hidden size for one-layer MLP DQN network: MLP_Q_Net
    net_extra_hyperparameters = {'hidden_size': hidden_size}
    evaluate_episodes_for_conv = EVALUATE_EPISODES_FOR_CONV
    solve_criterion = SOLVE_CRITERION
    replay_size = REPLAY_SIZE
    agent_trainer = double_DQN
    # train_episodes_upper_bound should be specified later on
    agent_trainer_hyperparameters = {'net_class': net_class,
                                     'net_extra_hyperparameters': net_extra_hyperparameters,
                                     'epsilon': init_epsilon,
                                     'epsilon_decay': epsilon_decay,
                                     'sync_interval': tgt_net_sync,
                                     'gamma': gamma,
                                     'batch_size': batch_size,
                                     'lr': lr,
                                     'evaluate_episodes_for_conv': evaluate_episodes_for_conv,
                                     'solve_criterion': solve_criterion,
                                     'replay_size': replay_size,
                                     'silent_flag': True}
    recorder_class = AbsoluteRewardRecorder
    show_interval = SHOW_INTERVAL
    evaluate_episodes_for_eval = EVALUATE_EPISODES_FOR_EVAL
    recorder_extra_hyperparameters = {'evaluate_episodes_for_eval': evaluate_episodes_for_eval}

    merge_interval_set = [1, 2]

    exp_diff_merge_interval(env_class=env_class,
                            net_class=net_class,
                            recorder_class=recorder_class,
                            agent_trainer=agent_trainer,
                            net_extra_hyperparameters=net_extra_hyperparameters,
                            recorder_extra_hyperparameters=recorder_extra_hyperparameters,
                            agent_trainer_hyperparameters=agent_trainer_hyperparameters,
                            merge_interval_set=merge_interval_set,
                            gamma=gamma,
                            train_episodes_upper_bound=train_episodes_upper_bound,
                            show_interval=show_interval)
    report_diff_merge_interval(env_class=env_class,
                               train_episodes_upper_bound=train_episodes_upper_bound,
                               train_size=train_size,
                               validation_size=validation_size,
                               gamma=gamma,
                               merge_interval_set=merge_interval_set)
    return


def exp_diff_merge_interval(env_class: Type[MyEnv],
                            net_class: Type[RLNet],
                            recorder_class: Type[ExperimentRecorder],
                            agent_trainer,
                            net_extra_hyperparameters: dict,
                            recorder_extra_hyperparameters: dict,
                            agent_trainer_hyperparameters: dict,
                            merge_interval_set: list,
                            gamma=GAMMA,
                            train_episodes_upper_bound=TRAIN_EPISODES_UB,
                            show_interval=SHOW_INTERVAL):
    # mkdir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    merge_interval_set_size = len(merge_interval_set)

    # Initialize train, validation, test theta set
    train_theta_set = np.load(log_dir + 'train_theta.npy')
    validation_theta_set = np.load(log_dir + 'validation_theta.npy')
    test_theta_set = np.load(log_dir + 'test_theta.npy')

    for count in tqdm(range(outer_loop_num)):
        # Train our algorithm with different merge interval
        for k in range(merge_interval_set_size):
            merge_interval = merge_interval_set[k]
            current_log_dir = log_dir + 'm=' + str(merge_interval) + 'out=' + str(count) + '/'
            # Call QAvg
            net, _, _, _ =\
                DQNAvg(net_class=net_class,
                       net_extra_hyperparameters=net_extra_hyperparameters,
                       env_class=env_class,
                       recorder_class=recorder_class,
                       recorder_extra_hyperparameters=recorder_extra_hyperparameters,
                       train_theta_set=train_theta_set[count],
                       validation_theta_set=validation_theta_set,
                       test_theta_set=test_theta_set,
                       agent_trainer=agent_trainer,
                       agent_trainer_hyperparameters=agent_trainer_hyperparameters,
                       gamma=gamma,
                       merge_interval=merge_interval,
                       show_interval=show_interval,
                       train_episodes_upper_bound=train_episodes_upper_bound,
                       log_dir=current_log_dir,
                       silent_flag=True)
    return


# We could write report function for recorder class, i.e. write this function by calling recorder class method
def report_diff_merge_interval(env_class: Type[MyEnv],
                               merge_interval_set: list,
                               train_episodes_upper_bound=TRAIN_EPISODES_UB,
                               train_size=TRAIN_SIZE,
                               validation_size=VALIDATION_SIZE,
                               gamma=GAMMA):
    min_merge_interval = min(merge_interval_set)
    report_size = train_episodes_upper_bound // min_merge_interval  # We assume this is an exact division
    train_theta_set = np.load(log_dir + 'train_theta.npy')
    validation_theta_set = np.load(log_dir + 'validation_theta.npy')

    total_cum_reward_E1 = np.zeros((outer_loop_num, report_size))
    total_cum_reward_E2 = np.zeros((outer_loop_num, report_size))
    total_cum_reward_EInf = np.zeros((outer_loop_num, report_size))
    total_cum_reward_base = np.zeros((outer_loop_num, report_size))
    total_cum_reward_base_avg = np.zeros((outer_loop_num, report_size))

    total_test_cum_reward_E1 = np.zeros((outer_loop_num, report_size))
    total_test_cum_reward_E2 = np.zeros((outer_loop_num, report_size))
    total_test_cum_reward_EInf = np.zeros((outer_loop_num, report_size))
    total_test_cum_reward_base = np.zeros((outer_loop_num, report_size))
    total_test_cum_reward_base_avg = np.zeros((outer_loop_num, report_size))

    for count in tqdm(range(outer_loop_num)):
        cum_reward_E1 = np.load(log_dir + f'm={merge_interval_set[0]}out={count}/train/cum_reward_avg.npy')
        test_cum_reward_E1 = np.load(log_dir + f'm={merge_interval_set[0]}out={count}/validation/cum_reward_avg.npy')
        tmp_cum_reward_E2 = np.load(log_dir + f'm={merge_interval_set[1]}out={count}/train/cum_reward_avg.npy')
        tmp_test_cum_reward_E2 = np.load(log_dir + f'm={merge_interval_set[1]}out={count}/validation/cum_reward_avg.npy')

        E12ratio = merge_interval_set[1] // merge_interval_set[0]
        cum_reward_E2 = np.zeros((report_size))
        test_cum_reward_E2 = np.zeros((report_size))
        for i in range(report_size // E12ratio):
            cum_reward_E2[i*E12ratio:(i+1)*E12ratio] = tmp_cum_reward_E2[i]
            test_cum_reward_E2[i*E12ratio:(i+1)*E12ratio] = tmp_test_cum_reward_E2[i]

        cum_reward_base = np.zeros((report_size))
        cum_reward_base_avg = np.zeros((report_size))
        cum_reward_EInf = np.zeros((report_size))
        test_cum_reward_base = np.zeros((report_size))
        test_cum_reward_base_avg = np.zeros((report_size))
        test_cum_reward_EInf = np.zeros((report_size))

        # The following codes are for Cart-Pole experiments
        env_train = []
        env_valid = []
        nets = []
        for theta in train_theta_set[count]:
            env_train.append(env_class(para=theta, mode='link_mass_1'))
        for theta in validation_theta_set:
            env_valid.append(env_class(para=theta, mode='link_mass_1'))
        current_init_epsilon = INIT_EPSILON
        epsilon_decay = EPS_DECAY
        net_extra_hyperparameters = {'hidden_size': HIDDEN_SIZE}
        agent_trainer_hyperparameters = {'net_class': MLP_Q_Net,
                                         'net_extra_hyperparameters': net_extra_hyperparameters,
                                         'epsilon': INIT_EPSILON,
                                         'epsilon_decay': EPS_DECAY,
                                         'sync_interval': TGT_NET_SYNC,
                                         'gamma': GAMMA,
                                         'batch_size': BATCH_SIZE,
                                         'lr': LR,
                                         'evaluate_episodes_for_conv': EVALUATE_EPISODES_FOR_CONV,
                                         'solve_criterion': SOLVE_CRITERION,
                                         'replay_size': REPLAY_SIZE,
                                         'train_episodes_upper_bound': min_merge_interval,
                                         'silent_flag': True}
        obs_size = env_train[0].observation_space.shape[0]
        n_actions = env_train[0].action_space.n
        net = MLP_Q_Net(obs_size=obs_size, n_actions=n_actions, **net_extra_hyperparameters)
        net = net.to(device)
        # Compute baseline-Max and baseline-Avg and Q-Avg with E=Inf
        for i in range(report_size):
            #########
            # Train #
            #########
            agent_trainer_hyperparameters['epsilon'] = current_init_epsilon
            current_init_epsilon *= (epsilon_decay ** min_merge_interval)
            if i == 0:
                for j in range(train_size):
                    tmp_net, _ = double_DQN(env=env_train[j],
                                            continue_train_flag=False,
                                            **agent_trainer_hyperparameters
                                            )
                    nets.append(tmp_net)
            else:
                for j in range(train_size):
                    tmp_net, _ = double_DQN(env=env_train[j],
                                            continue_train_flag=True,
                                            continue_train_para=nets[j].state_dict(),
                                            **agent_trainer_hyperparameters
                                            )
                    nets[j] = tmp_net
            # Compute E = Inf
            net_para = 0
            for j in range(train_size):
                tmp_para = nets[j].state_dict()
                if j == 0:
                    net_para = copy.deepcopy(tmp_para)
                else:
                    net_para = net_para_add(net_para, tmp_para)
            net_para = net_para_scale(net_para, (1 / train_size))
            net.load_state_dict(net_para)


            ################
            # Record error #
            ################
            env_cum_reward_base = np.zeros((train_size, train_size))
            env_test_cum_reward_base = np.zeros((train_size, validation_size))

            for j in range(train_size):
                # Record training error
                for k in range(train_size):
                    tmp_env = env_class(para=train_theta_set[count][k], mode='link_mass_1')
                    tmp_env.reset()
                    env_cum_reward_base[j][k] = \
                        nets[j].evaluate_expected_reward(env=tmp_env,
                                                         evaluate_episodes_for_eval=EVALUATE_EPISODES_FOR_EVAL,
                                                         gamma=gamma)
                # Record validation error
                for k in range(validation_size):
                    tmp_env = env_class(para=validation_theta_set[k], mode='link_mass_1')
                    tmp_env.reset()
                    env_test_cum_reward_base[j][k] = \
                        nets[j].evaluate_expected_reward(env=tmp_env,
                                                         evaluate_episodes_for_eval=EVALUATE_EPISODES_FOR_EVAL,
                                                         gamma=gamma)
            env_avg_cum_reward_base = np.average(env_cum_reward_base, axis=1)
            env_avg_test_cum_reward_base = np.average(env_test_cum_reward_base, axis=1)
            cum_reward_base[i] = np.max(env_avg_cum_reward_base)
            cum_reward_base_avg[i] = np.average(env_avg_cum_reward_base)
            test_cum_reward_base[i] = np.max(env_avg_test_cum_reward_base)
            test_cum_reward_base_avg[i] = np.average(env_avg_test_cum_reward_base)

            # Record train error for E=Inf
            EInf_cum_reward = np.zeros((train_size))
            EInf_test_cum_reward = np.zeros((validation_size))
            for k in range(train_size):
                tmp_env = env_class(para=train_theta_set[count][k], mode='link_mass_1')
                tmp_env.reset()
                EInf_cum_reward[k] = \
                    net.evaluate_expected_reward(env=tmp_env,
                                                 evaluate_episodes_for_eval=EVALUATE_EPISODES_FOR_EVAL,
                                                 gamma=gamma)
            # Record test error for E=Inf
            for k in range(validation_size):
                tmp_env = env_class(para=validation_theta_set[k], mode='link_mass_1')
                tmp_env.reset()
                EInf_test_cum_reward[k] = \
                    net.evaluate_expected_reward(env=tmp_env,
                                                 evaluate_episodes_for_eval=EVALUATE_EPISODES_FOR_EVAL,
                                                 gamma=gamma)
            cum_reward_EInf[i] = np.average(EInf_cum_reward)
            test_cum_reward_EInf[i] = np.average(EInf_test_cum_reward)
        total_cum_reward_E1[count] = cum_reward_E1
        total_cum_reward_E2[count] = cum_reward_E2
        total_cum_reward_EInf[count] = cum_reward_EInf
        total_cum_reward_base[count] = cum_reward_base
        total_cum_reward_base_avg[count] = cum_reward_base_avg

        total_test_cum_reward_E1[count] = test_cum_reward_E1
        total_test_cum_reward_E2[count] = test_cum_reward_E2
        total_test_cum_reward_EInf[count] = test_cum_reward_EInf
        total_test_cum_reward_base[count] = test_cum_reward_base
        total_test_cum_reward_base_avg[count] = test_cum_reward_base_avg

    if not os.path.exists('./' + save_dir):
        os.makedirs('./' + save_dir)

    np.save('./' + save_dir + '/MEOBj_lst.npy', total_cum_reward_E1)
    np.save('./' + save_dir + '/MEOBj2_lst.npy', total_cum_reward_EInf)
    np.save('./' + save_dir + '/MEOBj3_lst.npy', total_cum_reward_E2)
    np.save('./' + save_dir + '/MEOBj_base_lst.npy', total_cum_reward_base)
    np.save('./' + save_dir + '/MEOBj_base_avg_lst.npy', total_cum_reward_base_avg)

    np.save('./' + save_dir + '/test_MEOBj_lst.npy', total_test_cum_reward_E1)
    np.save('./' + save_dir + '/test_MEOBj2_lst.npy', total_test_cum_reward_EInf)
    np.save('./' + save_dir + '/test_MEOBj3_lst.npy', total_test_cum_reward_E2)
    np.save('./' + save_dir + '/test_MEOBj_base_lst.npy', total_test_cum_reward_base)
    np.save('./' + save_dir + '/test_MEOBj_base_avg_lst.npy', total_test_cum_reward_base_avg)
    return


if __name__ == '__main__':
    integrate_diff_merge_interval()
