import numpy as np
from DeepRLAlgo import RLNet
from MyEnv import MyEnv
from MyCartPole import MyCartPoleEnv
from typing import Type
from functools import wraps
from Config import *


# Compute optimal cumulative reward for an environment
# Since this function could be used in many places, we do not implement it as a method of the reporter class
def get_opt_cum_reward(env_class: Type[MyEnv], para):
    # In this function, you should specify how to solve an environment separately
    # Cuz there is no universal algorithm guaranteeing good performance for all reinforcement learning problems
    # For example, MountainCar is hard for double DQN algorithm, while not hard for noisy DQN
    if env_class == MyCartPoleEnv:
        opt_cum_reward = 200
    else:
        raise ValueError("Do not support this environment class")
    return opt_cum_reward


# Get the string of a parameterized environment, str = environment name + ' ' + str(para)
# We truncate the length of a string of a float number to at most 5, e.g. 0.123
def get_para_env_name(env_class: Type[MyEnv], para: float):
    truncate_length = 5
    env_name = env_class.__name__ + ' ' + (str(para)[:truncate_length+1])
    return env_name


# Wrapper for class method: record
def record_wrapper(record_method):
    @wraps(record_method)
    def wrapRecordMethod(self, **kwargs):
        if self.done:
            print('The recorder is full, do not call record any more.')
            return
        # Call record
        record_method(self, **kwargs)
        # Update self.current_step and self.done
        self.current_step += 1
        if self.current_step == self.record_len:
            self.done = True
            self.write_log()
        return
    return wrapRecordMethod


# Wrapper for class method: report
def report_wrapper(report_method):
    @wraps(report_method)
    def wrapReportMethod(self, **kwargs):
        if self.current_step == 0:
            print('The recorder is empty, please call report later.')
            return
        print(CUTTING_LINE)
        print(self.recorder_name + ': Current step is ' + str(self.current_step) + '/' + str(self.record_len))
        # Call report
        report_method(self, **kwargs)
        return
    return wrapReportMethod


# Wrapper for class method: get full report
def get_full_report_wrapper(gfr_method):
    @wraps(gfr_method)
    def wrapGFRMethod(self):
        if not self.done:
            print('Warning: the recorder is not full, and you call get_full_report.')
        # Call and return class method
        return gfr_method(self)
    return wrapGFRMethod


class ExperimentRecorder:
    def __init__(self,
                 env_class: Type[MyEnv],
                 theta_set: np.ndarray,
                 merge_interval: int,
                 log_folder_dir: str,
                 record_len: int,
                 recorder_name: str,
                 gamma: float,
                 **kwargs):
        self.recorder_type_list = None   # This is just for avoid warning
        self.env_class = env_class
        self.theta_set = theta_set
        self.merge_interval = merge_interval
        self.log_folder_dir = log_folder_dir
        self.record_len = record_len
        self.recorder_name = recorder_name
        self.gamma = gamma

        self.env_size = self.theta_set.size

        self.current_step = 0
        self.done = False

    @record_wrapper
    def record(self, net: RLNet, **kwargs):
        # Do nothing
        return

    @report_wrapper
    def report(self):
        # Do nothing
        return

    # This method is silent, and usually return an ndarray
    @get_full_report_wrapper
    def get_full_report(self):
        # Do nothing, but must return a list
        return []

    # This method is called only in record method, do not call it by yourself
    # We do not use private method here for convenience, but do not call it by yourself
    def write_log(self):
        # Do nothing
        return


class CombineRecorder(ExperimentRecorder):
    def __init__(self,
                 env_class: Type[MyEnv],
                 theta_set: np.ndarray,
                 merge_interval: int,
                 log_folder_dir: str,
                 record_len: int,
                 recorder_name: str,
                 gamma: float,
                 recorder_type_list: list,
                 **kwargs):
        super(CombineRecorder, self).__init__(env_class=env_class,
                                              theta_set=theta_set,
                                              merge_interval=merge_interval,
                                              log_folder_dir=log_folder_dir,
                                              record_len=record_len,
                                              recorder_name=recorder_name,
                                              gamma=gamma)
        self.recorder_type_list = recorder_type_list
        self.recorder_list = []
        # Examine recorder_list
        for recorder_type in self.recorder_type_list:
            assert issubclass(recorder_type, ExperimentRecorder)
            recorder = recorder_type(env_class=env_class,
                                     theta_set=theta_set,
                                     merge_interval=merge_interval,
                                     log_folder_dir=log_folder_dir,
                                     record_len=record_len,
                                     recorder_name=recorder_name,
                                     gamma=gamma,
                                     **kwargs)
            self.recorder_list.append(recorder)

    @record_wrapper
    def record(self, net: RLNet, **kwargs):
        for recorder in self.recorder_list:
            recorder.record(net=net, **kwargs)
        return

    @report_wrapper
    def report(self):
        for recorder in self.recorder_list:
            # Only the outermost wrapper is required
            recorder.report.__wrapped__(recorder)
        return

    @get_full_report_wrapper
    def get_full_report(self):
        result = []
        for recorder in self.recorder_list:
            # We assume get_full_report must return a list
            result.append(recorder.get_full_report())
        return result

    def write_log(self):
        # every write_log has been called during call every recorder's record, thus this function just do nothing
        return


# We usually cannot compute the exact error, thus we use a surrogate error:
# average differences between best agent in every environment
class SurrogateErrorRecorder(ExperimentRecorder):
    def __init__(self,
                 env_class: Type[MyEnv],
                 theta_set: np.ndarray,
                 merge_interval: int,
                 log_folder_dir: str,
                 record_len: int,
                 recorder_name: str,
                 gamma: float,
                 evaluate_episodes_for_eval: int,
                 **kwargs):
        super(SurrogateErrorRecorder, self).__init__(env_class=env_class,
                                                     theta_set=theta_set,
                                                     merge_interval=merge_interval,
                                                     log_folder_dir=log_folder_dir,
                                                     record_len=record_len,
                                                     recorder_name=recorder_name,
                                                     gamma=gamma)
        self.evaluate_episodes_for_eval = evaluate_episodes_for_eval
        self.opt_cum_reward = (np.ones(self.env_size) * (-1.0))  # Initialize it with -1, to make detecting error easy
        self.compute_opt_cum_reward()
        self.cum_reward_error = np.zeros((self.record_len, self.env_size))
        self.cum_reward_avg_error = np.zeros(self.record_len)

    # Maintain this static dictionary to avoid repetitive computation
    opt_cum_reward_dict = {}

    # Call this class method after you define a series of new environments
    # e.g. in an exp function, call it after you define theta_train_set and theta_validation_set, before you call QAvg
    @classmethod
    def init_opt_cum_reward_dict(cls):
        cls.opt_cum_reward_dict = {}
        return

    def compute_opt_cum_reward(self):
        for i in range(self.env_size):
            env_name = get_para_env_name(env_class=self.env_class, para=self.theta_set[i])
            # Avoid repetitive computation: save the opt_cum_reward we have already computed
            if env_name in SurrogateErrorRecorder.opt_cum_reward_dict:
                self.opt_cum_reward[i] = SurrogateErrorRecorder.opt_cum_reward_dict[env_name]
            else:
                self.opt_cum_reward[i] = get_opt_cum_reward(env_class=self.env_class, para=self.theta_set[i])
                SurrogateErrorRecorder.opt_cum_reward_dict[env_name] = self.opt_cum_reward[i]
        return

    @record_wrapper
    def record(self, net: RLNet, **kwargs):
        for i in range(self.env_size):
            tmp_env = self.env_class(para=self.theta_set[i])
            self.cum_reward_error[self.current_step][i] = \
                self.opt_cum_reward[i] - \
                net.evaluate_expected_reward(env=tmp_env,
                                             evaluate_episodes_for_eval=self.evaluate_episodes_for_eval,
                                             gamma=self.gamma)
        self.cum_reward_avg_error[self.current_step] = np.average(self.cum_reward_error[self.current_step])
        return

    @report_wrapper
    def report(self):
        print('Most recent average surrogate cumulative reward error: ' +
              str(self.cum_reward_avg_error[self.current_step - 1]))
        return

    @get_full_report_wrapper
    def get_full_report(self):
        return self.cum_reward_error, self.cum_reward_avg_error

    def write_log(self):
        # Save the dictionary, you can load it by:
        # opt_cum_reward_dict = np.load(path, allow_pickle=True).item(), then you will get the original dictionary
        np.save(self.log_folder_dir + 'opt_cum_reward_dict.npy',
                SurrogateErrorRecorder.opt_cum_reward_dict, allow_pickle=True)
        # We save the dictionary as well as the array, use whatever you like
        np.save(self.log_folder_dir + 'opt_cum_reward.npy', self.opt_cum_reward)
        np.save(self.log_folder_dir + 'cum_reward_surrogate_error.npy', self.cum_reward_error)
        np.save(self.log_folder_dir + 'cum_reward_avg_surrogate_error.npy', self.cum_reward_avg_error)
        return


# This is a simpler recorder, which record the average absolute cumulative reward
class AbsoluteRewardRecorder(ExperimentRecorder):
    def __init__(self,
                 env_class: Type[MyEnv],
                 theta_set: np.ndarray,
                 merge_interval: int,
                 log_folder_dir: str,
                 record_len: int,
                 recorder_name: str,
                 gamma: float,
                 evaluate_episodes_for_eval: int,
                 **kwargs):
        super(AbsoluteRewardRecorder, self).__init__(env_class=env_class,
                                                     theta_set=theta_set,
                                                     merge_interval=merge_interval,
                                                     log_folder_dir=log_folder_dir,
                                                     record_len=record_len,
                                                     recorder_name=recorder_name,
                                                     gamma=gamma)
        self.evaluate_episodes_for_eval = evaluate_episodes_for_eval
        self.cum_reward = np.zeros((self.record_len, self.env_size))
        self.cum_reward_avg = np.zeros(self.record_len)

    @record_wrapper
    def record(self, net: RLNet, **kwargs):
        for i in range(self.env_size):
            tmp_env = self.env_class(para=self.theta_set[i])
            self.cum_reward[self.current_step][i] = \
                net.evaluate_expected_reward(env=tmp_env,
                                             evaluate_episodes_for_eval=self.evaluate_episodes_for_eval,
                                             gamma=self.gamma)
        self.cum_reward_avg[self.current_step] = np.average(self.cum_reward[self.current_step])
        return

    @report_wrapper
    def report(self):
        print('Most recent average cumulative reward: ' + str(self.cum_reward_avg[self.current_step-1]))
        return

    @get_full_report_wrapper
    def get_full_report(self):
        if not self.done:
            print('Warning: the recorder is not full, and you call get_full_report.')
        return self.cum_reward, self.cum_reward_avg

    def write_log(self):
        np.save(self.log_folder_dir + 'cum_reward.npy', self.cum_reward)
        np.save(self.log_folder_dir + 'cum_reward_avg.npy', self.cum_reward_avg)
        return
