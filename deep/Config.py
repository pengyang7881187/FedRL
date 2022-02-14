import torch

# device
device = torch.device("cuda")

# Random seed
SEED = 666

# Default hyperparameters
GAMMA = 0.9
BATCH_SIZE = 16
HIDDEN_SIZE = 128
TGT_NET_SYNC = 10  # Synchronization interval for double DQN
REPLAY_SIZE = 1000
LR = 1e-3  # Learning rate for DQN algorithm
EPS_DECAY = 0.99  # This is for epsilon-greedy exploration strategy
EVALUATE_EPISODES_FOR_CONV = 5  # This is for judging convergence of an algorithm
EVALUATE_EPISODES_FOR_EVAL = 5  # This is for evaluating reward of a policy on an environment

SOLVE_CRITERION = 199  # Default for CartPole and Acrobot
INIT_EPSILON = 1
TRAIN_EPISODES_UB = 384  # 3 * 128

MERGE_INTERVAL = 16  # This interval is for inner loop, i.e. episodes interval
SHOW_INTERVAL = 1  # This interval is for outer loop

CUTTING_LINE_LEN = 60
CUTTING_LINE = '-' * CUTTING_LINE_LEN

TRAIN_SIZE = 5
VALIDATION_SIZE = 20
TEST_SIZE = 10

# For experiment: exp_diff_merge_interval
MERGE_INTERVAL_SET_SIZE = 5  # We assume TRAIN_EPISODES_UB % (2 ^ MERGE_INTERVAL_SET_SIZE) = 0
DIFF_MERGE_LOG_DIR = './exp-differ-merge-interval/'

