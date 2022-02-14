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


def DDPGAvg(env_class: Type[MyEnv],
            recorder_class: Type[ExperimentRecorder],
            recorder_extra_hyperparameters: dict,  # Do not parse None to it!
            train_theta_set: np.ndarray,
            validation_theta_set: np.ndarray,
            test_theta_set: np.ndarray,
            agent_trainer=DDPG_TRAIN,
            gamma=0.99,
            merge_interval=MERGE_INTERVAL,
            total_frame_upper_bound=int(1e+6),
            log_dir='./log-pendulum/',
            crt_flag=True,
            silent_flag=True,
            threshold=2.0):

    # mkdir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        os.makedirs(log_dir + 'train/')
        os.makedirs(log_dir + 'validation/')
        os.makedirs(log_dir + 'test/')

    assert total_frame_upper_bound % merge_interval == 0  # This assumption is for convenience

    # Prepare environments and set some parameters
    # We do not need validation_size and test_size in this function
    train_size = train_theta_set.size
    env_train = []
    for theta in train_theta_set:
        env_train.append(env_class(para=theta))
    obs_size = env_train[0].observation_space.shape[0]
    action_size = env_train[0].action_space.shape[0]
    # n_actions = env_train[0].action_space.n

    # Prepare recorder
    # Outer loop: train agents and merge their parameters
    outer_loop_iter_upper_bound = int(total_frame_upper_bound / merge_interval)
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
                                   merge_interval=total_frame_upper_bound,
                                   log_folder_dir=test_recorder_log_dir,
                                   record_len=1,
                                   recorder_name=test_recorder_name,
                                   gamma=gamma,
                                   **recorder_extra_hyperparameters)

    # Initialize network or load network parameter
    act_net = DDPGActor(
        env_train[0].observation_space.shape[0],
        env_train[0].action_space.shape[0],
        threshold).to(device)
    crt_net = DDPGCritic(
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
    # Test our net
    test_recorder.record(net=act_net)
    if not silent_flag:
        # Report test error
        test_recorder.report()
    return act_net, train_recorder, validation_recorder, test_recorder
