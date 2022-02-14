import copy
import argparse
from DDPGAvg import net_para_add, net_para_scale, DDPGAvg
from DeepRLAlgo import *
from ExpRecorder import ExperimentRecorder, AbsoluteRewardRecorder
from typing import Type
from MyHalfCheetah import MyHalfCheetahEnv
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default=0, type=int)

args = parser.parse_args()
gpu = args.gpu

os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

log_dir = './DDPGAvg-HalfCheetah/'
save_dir = 'plot-DDPGAvg-HalfCheetah'
outer_loop_num = 100
threshold = 1.0  # This parameter is only for HalfCheetah


# We separate train and report, cuz train is very time-consuming, while report is not
def integrate_diff_merge_interval():
    # mkdir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    seed2 = 250
    np.random.seed(seed2)

    # Prepare some parameters: most are loaded from Config.py
    env_class = MyHalfCheetahEnv
    gamma = 0.99
    total_frame_upper_bound = int(3200000)   # 320000

    train_size = 5
    validation_size = 5
    test_size = 2

    train_theta_set = np.random.uniform(size=(outer_loop_num, train_size))
    validation_theta_set = np.linspace(0, 1, validation_size)
    test_theta_set = np.random.uniform(size=test_size)
    np.save(log_dir + 'train_theta.npy', train_theta_set)
    np.save(log_dir + 'validation_theta.npy', validation_theta_set)
    np.save(log_dir + 'test_theta.npy', test_theta_set)

    agent_trainer = DDPG_TRAIN

    recorder_class = AbsoluteRewardRecorder
    evaluate_episodes_for_eval = EVALUATE_EPISODES_FOR_EVAL
    recorder_extra_hyperparameters = {'evaluate_episodes_for_eval': evaluate_episodes_for_eval}

    merge_interval_set = [20000, 40000]

    exp_diff_merge_interval(env_class=env_class,
                            recorder_class=recorder_class,
                            agent_trainer=agent_trainer,
                            recorder_extra_hyperparameters=recorder_extra_hyperparameters,
                            merge_interval_set=merge_interval_set,
                            gamma=gamma,
                            total_frame_upper_bound=total_frame_upper_bound)
    report_diff_merge_interval(env_class=env_class,
                               total_frame_upper_bound=total_frame_upper_bound,
                               train_size=train_size,
                               validation_size=validation_size,
                               gamma=gamma,
                               merge_interval_set=merge_interval_set)
    return


def exp_diff_merge_interval(env_class: Type[MyEnv],
                            recorder_class: Type[ExperimentRecorder],
                            agent_trainer,
                            recorder_extra_hyperparameters: dict,
                            merge_interval_set: list,
                            gamma=GAMMA,
                            total_frame_upper_bound=int(1e+6)):

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
            # Call DDPG Avg
            act_net, _, _, _ =\
                DDPGAvg(env_class=env_class,
                        recorder_class=recorder_class,
                        recorder_extra_hyperparameters=recorder_extra_hyperparameters,
                        train_theta_set=train_theta_set[count],
                        validation_theta_set=validation_theta_set,
                        test_theta_set=test_theta_set,
                        agent_trainer=agent_trainer,
                        gamma=gamma,
                        merge_interval=merge_interval,
                        total_frame_upper_bound=total_frame_upper_bound,
                        log_dir=current_log_dir,
                        crt_flag=True,
                        silent_flag=False,
                        threshold=threshold)
    return


def report_diff_merge_interval(env_class: Type[MyEnv],
                               merge_interval_set: list,
                               total_frame_upper_bound=int(1e+6),
                               train_size=TRAIN_SIZE,
                               validation_size=VALIDATION_SIZE,
                               gamma=GAMMA):
    min_merge_interval = min(merge_interval_set)
    report_size = total_frame_upper_bound // min_merge_interval  # We assume this is an exact division
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

        # The following codes are for Pendulum experiments
        env_train = []
        env_valid = []
        act_nets = []
        crt_nets = []
        for theta in train_theta_set[count]:
            env_train.append(env_class(para=theta))
        for theta in validation_theta_set:
            env_valid.append(env_class(para=theta))

        act_net = DDPGActor(
            env_train[0].observation_space.shape[0],
            env_train[0].action_space.shape[0],
            threshold).to(device)
        # Compute baseline-Max and baseline-Avg and DDPGAvg with E=Inf
        for i in range(report_size):
            #########
            # Train #
            #########
            if i == 0:
                for j in range(train_size):
                    tmp_act_net, tmp_crt_net, _ = DDPG_TRAIN(env=env_train[j], frame_upper_bound=min_merge_interval,
                                                             threshold=threshold)
                    act_nets.append(tmp_act_net)
                    crt_nets.append(tmp_crt_net)
            else:
                for j in range(train_size):
                    tmp_act_net, tmp_crt_net, _ = DDPG_TRAIN(env=env_train[j],
                                                             threshold=threshold,
                                                             frame_upper_bound=min_merge_interval,
                                                             continue_train_flag=True,
                                                             continue_train_act_para=act_nets[j].state_dict(),
                                                             continue_train_crt_para=crt_nets[j].state_dict())
                    act_nets[j] = tmp_act_net
                    crt_nets[j] = tmp_crt_net
            # Compute E = Inf
            act_net_para = 0
            for j in range(train_size):
                tmp_act_para = act_nets[j].state_dict()
                if j == 0:
                    act_net_para = copy.deepcopy(tmp_act_para)
                else:
                    act_net_para = net_para_add(act_net_para, tmp_act_para)
            act_net_para = net_para_scale(act_net_para, (1 / train_size))
            act_net.load_state_dict(act_net_para)


            ################
            # Record error #
            ################
            env_cum_reward_base = np.zeros((train_size, train_size))
            env_test_cum_reward_base = np.zeros((train_size, validation_size))

            for j in range(train_size):
                # Record training error
                for k in range(train_size):
                    tmp_env = env_class(para=train_theta_set[count][k])
                    tmp_env.reset()
                    env_cum_reward_base[j][k] = \
                        act_nets[j].evaluate_expected_reward(env=tmp_env,
                                                             threshold=threshold,
                                                             evaluate_episodes_for_eval=EVALUATE_EPISODES_FOR_EVAL,
                                                             gamma=gamma)
                # Record validation error
                for k in range(validation_size):
                    tmp_env = env_class(para=validation_theta_set[k])
                    tmp_env.reset()
                    env_test_cum_reward_base[j][k] = \
                        act_nets[j].evaluate_expected_reward(env=tmp_env,
                                                             threshold=threshold,
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
                tmp_env = env_class(para=train_theta_set[count][k])
                tmp_env.reset()
                EInf_cum_reward[k] = \
                    act_net.evaluate_expected_reward(env=tmp_env,
                                                     threshold=threshold,
                                                     evaluate_episodes_for_eval=EVALUATE_EPISODES_FOR_EVAL,
                                                     gamma=gamma)
            # Record test error for E=Inf
            for k in range(validation_size):
                tmp_env = env_class(para=validation_theta_set[k])
                tmp_env.reset()
                EInf_test_cum_reward[k] = \
                    act_net.evaluate_expected_reward(env=tmp_env,
                                                     threshold=threshold,
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

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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
