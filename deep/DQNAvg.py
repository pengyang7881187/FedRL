import copy
from DeepRLAlgo import *
from ExpRecorder import ExperimentRecorder
from typing import Type


def net_para_add(para_a, para_b):
    para_c = copy.deepcopy(para_a)
    for key in para_c:
        para_c[key] = para_a[key] + para_b[key]
    return para_c


def net_para_scale(para_a, scale):
    para_c = copy.deepcopy(para_a)
    for key in para_c:
        para_c[key] = para_a[key] * scale
    return para_c


def DQNAvg(net_class: Type[RLNet],
           net_extra_hyperparameters: dict,  # Do not parse None to it!
           env_class: Type[MyEnv],
           recorder_class: Type[ExperimentRecorder],
           recorder_extra_hyperparameters: dict,  # Do not parse None to it!
           train_theta_set: np.ndarray,
           validation_theta_set: np.ndarray,
           test_theta_set: np.ndarray,
           agent_trainer_hyperparameters: dict,  # Do not parse None to it!
           agent_trainer=double_DQN,
           gamma=GAMMA,
           merge_interval=MERGE_INTERVAL,
           show_interval=SHOW_INTERVAL,
           train_episodes_upper_bound=TRAIN_EPISODES_UB,
           continue_train_flag=False,
           continue_train_para=None,
           exploration_strategy='EPSILON_GREEDY',
           log_dir='./log/',
           silent_flag=True):

    # mkdir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        os.makedirs(log_dir + 'train/')
        os.makedirs(log_dir + 'validation/')
        os.makedirs(log_dir + 'test/')

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
    test_recorder_log_dir = log_dir + 'test/'
    train_recorder_name = 'Train recorder'
    validation_recorder_name = 'Validation recorder'
    test_recorder_name = 'Test recorder'
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
    # Test recorder is different from the previous two recorders
    test_recorder = recorder_class(env_class=env_class,
                                   theta_set=test_theta_set,
                                   merge_interval=train_episodes_upper_bound,
                                   log_folder_dir=test_recorder_log_dir,
                                   record_len=1,
                                   recorder_name=test_recorder_name,
                                   gamma=gamma,
                                   **recorder_extra_hyperparameters)

    # Initialize network or load network parameter
    net = net_class(obs_size=obs_size, n_actions=n_actions, **net_extra_hyperparameters)
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
                tmp_net, _ = agent_trainer(env=env_train[j], **agent_trainer_hyperparameters)
                tmp_para = tmp_net.state_dict()
                if j == 0:
                    net_para = copy.deepcopy(tmp_para)
                else:
                    net_para = net_para_add(net_para, tmp_para)
        # Other rounds of iteration, continue train from net
        else:
            for j in range(train_size):
                tmp_net, _ = agent_trainer(env=env_train[j],
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
    # Test our net
    test_recorder.record(net=net)
    if not silent_flag:
        # Report test error
        test_recorder.report()
    return net, train_recorder, validation_recorder, test_recorder
