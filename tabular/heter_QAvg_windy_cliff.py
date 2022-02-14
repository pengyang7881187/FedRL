import os
import argparse
import matplotlib as mpl

from utils import *
from tqdm import tqdm
from GridWorldEnvironment import WindyCliffGridWorld

mpl.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument("--lengthX", default=4, type=int)
parser.add_argument("--lengthY", default=4, type=int)
parser.add_argument("--gamma", default=0.95, type=float)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--lr_decay", default=0.1, type=float)
parser.add_argument("--n", default=5, type=int)
parser.add_argument("--iter", default=640)
args = parser.parse_args()

lengthX, lengthY, gamma = args.lengthX, args.lengthY, args.gamma
ntrain = args.n
ntest = 2 * args.n

nS = lengthX * lengthY
nA = 4

d_0 = np.zeros(nS)
d_0[0] = 1.0

outer_iter_num = 16000

Seed = 105

E_lst = (1, 2, 4, 8, 16, 32)
eps_lst = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

E_size = len(E_lst) + 1
eps_lst_len = len(eps_lst)

separation_len = 50

save_dir = './heter-plot-QAvg-windy_cliff'


def experiment():
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    env_example = WindyCliffGridWorld(length_X=lengthX, length_Y=lengthY, gamma=gamma, theta=0)
    reward = env_example.state_reward.reshape(-1)
    R = np.expand_dims(reward, 1).repeat(nA, axis=1)

    np.random.seed(Seed)
    seeds = np.random.randint(low=0, high=1000000, size=outer_iter_num)

    total_MEObj_lst = np.zeros((outer_iter_num, eps_lst_len, E_size, args.iter))
    total_test_MEObj_lst = np.zeros((outer_iter_num, eps_lst_len, E_size, args.iter))

    total_test_center_MEObj_lst = np.zeros((outer_iter_num, eps_lst_len, E_size, args.iter))
    total_test_group_MEObj_lst = np.zeros((outer_iter_num, eps_lst_len, E_size, args.iter))

    total_MEObj_base_lst = np.zeros((outer_iter_num, eps_lst_len, args.iter))  # Baseline
    total_MEObj_base_avg_lst = np.zeros((outer_iter_num, eps_lst_len, args.iter))  # Baseline2
    total_test_MEObj_base_lst = np.zeros((outer_iter_num, eps_lst_len, args.iter))  # Baseline
    total_test_MEObj_base_avg_lst = np.zeros((outer_iter_num, eps_lst_len, args.iter))  # Baseline2

    for count in tqdm(range(outer_iter_num)):
        seed = seeds[count]
        np.random.seed(seed)
        Q_init = np.random.uniform(size=(nS, nA))

        mix_Ps = generate_mix_windycliff_Ps(n=ntrain + ntest, length_X=lengthX, length_Y=lengthY, eps_lst=eps_lst, seed=seed)

        for eps_index in range(eps_lst_len):
            Ps = mix_Ps[eps_index]

            train_P = Ps[:ntrain]
            test_P = Ps[ntrain:]

            center_P = [Ps[0]]
            group_P = generate_group_windycliff_Ps(length_X=lengthX, length_Y=lengthY)

            MEObj_lst = np.zeros((E_size, args.iter))
            test_MEObj_lst = np.zeros((E_size, args.iter))

            test_center_MEObj_lst = np.zeros((E_size, args.iter))
            test_group_MEObj_lst = np.zeros((E_size, args.iter))

            MEObj_base_lst = np.zeros((args.iter))  # Baseline
            MEObj_base_avg_lst = np.zeros((args.iter))  # Baseline2
            test_MEObj_base_lst = np.zeros((args.iter))  # Baseline
            test_MEObj_base_avg_lst = np.zeros((args.iter))  # Baseline2

            # QAvg with different Es
            for E_num in range(E_size-1):
                E = E_lst[E_num]
                Q = Q_init.copy()
                lr = args.lr
                for e in range(args.iter // E):
                    # If the decrease is too small or negative, reduce the lr
                    if e > 2 and (MEObj_lst[E_num][(e - 1)*E] - MEObj_lst[E_num][(e - 2)*E]) < MEObj_lst[E_num][(e - 2)*E] * 1e-3:
                        lr = lr * args.lr_decay

                    V = QtoV(Q)
                    pi = QtoPolicy(Q)

                    center_ME_Obj = evaluate_MEObj_from_policy(pi, R, center_P, d_0, gamma)
                    group_ME_Obj = evaluate_MEObj_from_policy(pi, R, group_P, d_0, gamma)

                    ME_Obj = evaluate_MEObj_from_policy(pi, R, train_P, d_0, gamma)
                    ME_Obj_test = evaluate_MEObj_from_policy(pi, R, test_P, d_0, gamma)
                    for ttt in range(E):
                        MEObj_lst[E_num][e * E + ttt] = ME_Obj
                        test_MEObj_lst[E_num][e * E + ttt] = ME_Obj_test

                        test_center_MEObj_lst[E_num][e * E + ttt] = center_ME_Obj
                        test_group_MEObj_lst[E_num][e * E + ttt] = group_ME_Obj

                    Qs = []
                    for i in range(ntrain):
                        Vi = V.copy()
                        for _ in range(E):
                            delta_Vi, _ = value_iteration(Vi, train_P[i], R, gamma)
                            Vi = (1 - lr) * Vi + lr * delta_Vi
                        Qi = VtoQ(Vi, train_P[i], R, gamma)
                        Qs.append(Qi)
                    Q = sum(Qs) / ntrain

            # Baseline: Train every agent separately, i.e. do not merge
            lr = args.lr
            Q = Q_init.copy()
            Qs = [Q.copy() for _ in range(ntrain)]
            Q_avg = Q
            for e in range(args.iter):
                if e > 2 and (MEObj_lst[-1][e - 1] - MEObj_lst[-1][e - 2]) < MEObj_lst[-1][e - 2] * 1e-3:
                    lr = lr * args.lr_decay

                pi_avg = QtoPolicy(Q_avg)

                ME_Obj = evaluate_MEObj_from_policy(pi_avg, R, train_P, d_0, gamma)
                ME_Obj_test = evaluate_MEObj_from_policy(pi_avg, R, test_P, d_0, gamma)

                center_ME_Obj = evaluate_MEObj_from_policy(pi_avg, R, center_P, d_0, gamma)
                group_ME_Obj = evaluate_MEObj_from_policy(pi_avg, R, group_P, d_0, gamma)

                MEObj_lst[-1][e] = ME_Obj
                test_MEObj_lst[-1][e] = ME_Obj_test

                test_center_MEObj_lst[-1][e] = center_ME_Obj
                test_group_MEObj_lst[-1][e] = group_ME_Obj


                for i in range(ntrain):
                    Qi = Qs[i]
                    Vi = QtoV(Qi)
                    delta_Vi, _ = value_iteration(Vi, train_P[i], R, gamma)
                    Vi = (1 - lr) * Vi + lr * delta_Vi
                    Qi = VtoQ(Vi, train_P[i], R, gamma)
                    Qs[i] = Qi
                Q_avg = sum(Qs) / ntrain

                best_ME_Obj = -inf
                best_label = -1
                avg_ME_Obj = 0
                avg_ME_Obj_test = 0
                ME_Obj_tests = []
                for i in range(ntrain):
                    Qi = Qs[i]
                    pi_i = QtoPolicy(Qi)
                    ME_Obj_i = evaluate_MEObj_from_policy(pi_i, R, train_P, d_0, gamma)
                    avg_ME_Obj += ME_Obj_i
                    if ME_Obj_i > best_ME_Obj:
                        best_ME_Obj = ME_Obj_i
                        best_label = i
                    # Test
                    ME_Obj_i_test = evaluate_MEObj_from_policy(pi_i, R, test_P, d_0, gamma)
                    avg_ME_Obj_test += ME_Obj_i_test
                    ME_Obj_tests.append(ME_Obj_i_test)
                MEObj_base_lst[e] = best_ME_Obj
                MEObj_base_avg_lst[e] = avg_ME_Obj / ntrain
                test_MEObj_base_avg_lst[e] = avg_ME_Obj_test / ntrain
                test_MEObj_base_lst[e] = ME_Obj_tests[best_label]

            total_MEObj_lst[count][eps_index] = MEObj_lst
            total_MEObj_base_lst[count][eps_index] = MEObj_base_lst
            total_MEObj_base_avg_lst[count][eps_index] = MEObj_base_avg_lst
            total_test_MEObj_lst[count][eps_index] = test_MEObj_lst
            total_test_MEObj_base_lst[count][eps_index] = test_MEObj_base_lst
            total_test_MEObj_base_avg_lst[count][eps_index] = test_MEObj_base_avg_lst

            total_test_center_MEObj_lst[count][eps_index] = test_center_MEObj_lst
            total_test_group_MEObj_lst[count][eps_index] = test_group_MEObj_lst

    np.save(save_dir + '/MEOBj_lst.npy', total_MEObj_lst)
    np.save(save_dir + '/MEOBj_base_lst.npy', total_MEObj_base_lst)
    np.save(save_dir + '/MEOBj_base_avg_lst.npy', total_MEObj_base_avg_lst)

    np.save(save_dir + '/test_MEOBj_lst.npy', total_test_MEObj_lst)
    np.save(save_dir + '/test_MEOBj_base_lst.npy', total_test_MEObj_base_lst)
    np.save(save_dir + '/test_MEOBj_base_avg_lst.npy', total_test_MEObj_base_avg_lst)

    np.save(save_dir + '/test_center_MEOBj_lst.npy', total_test_center_MEObj_lst)
    np.save(save_dir + '/test_group_MEOBj_lst.npy', total_test_group_MEObj_lst)
    return


def report():
    print('Save dir=' + save_dir)
    coff = np.sqrt(outer_iter_num)
    total_center_MEOBj_lst = np.load(save_dir + '/test_center_MEOBj_lst.npy')
    for E_num in range(E_size):
        print('-' * separation_len)
        if E_num == E_size - 1:
            print('E=Inf:')
        else:
            print(f'E={E_lst[E_num]}:')
        center_MEOBj_lst = total_center_MEOBj_lst[:, :, E_num, :]
        center_MEObj_mean_lst = []
        center_MEObj_std_lst = []
        for eps_index in range(eps_lst_len):
            center_MEObj_mean_lst.append(np.average(center_MEOBj_lst[:, eps_index, :], axis=0))
            center_MEObj_std_lst.append(np.std(center_MEOBj_lst[:, eps_index, :], axis=0) / coff)
        # Here, epsilon is kappa in the paper
        for eps_index in range(eps_lst_len):
            print(f'Center kappa={eps_lst[eps_index]}: Mean={center_MEObj_mean_lst[eps_index][-1]}, '
                  f'Std={center_MEObj_std_lst[eps_index][-1]}')
    return


if __name__ == '__main__':
    experiment()
    report()
