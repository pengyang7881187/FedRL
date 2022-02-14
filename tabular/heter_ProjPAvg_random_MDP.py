import os
import argparse
import matplotlib as mpl

from utils import *
from tqdm import tqdm

mpl.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument("--nS", default=5, type=int)
parser.add_argument("--nA", default=5, type=int)
parser.add_argument("--gamma", default=0.9, type=float)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--lr_decay", default=0.9, type=float)
parser.add_argument("--n", default=5, type=int)
parser.add_argument("--iter", default=960)
args = parser.parse_args()

nS, nA, gamma = args.nS, args.nA, args.gamma
ntrain = args.n
ntest = 4 * ntrain

d_0 = np.ones(nS)/nS

save_dir = './heter-plot-ProjPAvg-random_MDP'

outer_iter_num = 16000

Seed = 100

E_lst = (1, 2, 4, 8, 16, 32)
eps_lst = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

E_size = len(E_lst) + 1
eps_lst_len = len(eps_lst)

separation_len = 50


def experiment():
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.random.seed(Seed)
    seeds = np.random.randint(low=0, high=1000000, size=outer_iter_num)

    total_MEObj_lst = np.zeros((outer_iter_num, eps_lst_len, E_size, args.iter))
    total_test_MEObj_lst = np.zeros((outer_iter_num, eps_lst_len, E_size, args.iter))

    total_test_center_MEObj_lst = np.zeros((outer_iter_num, eps_lst_len, E_size, args.iter))

    total_MEObj_base_lst = np.zeros((outer_iter_num, eps_lst_len, args.iter))  # Baseline
    total_MEObj_base_avg_lst = np.zeros((outer_iter_num, eps_lst_len, args.iter))  # Baseline2
    total_test_MEObj_base_lst = np.zeros((outer_iter_num, eps_lst_len, args.iter))  # Baseline
    total_test_MEObj_base_avg_lst = np.zeros((outer_iter_num, eps_lst_len, args.iter))  # Baseline2

    for count in tqdm(range(outer_iter_num)):
        seed = seeds[count]
        R = np.random.uniform(low=0, high=1, size=(nS, nA))
        pi_init = np.random.uniform(size=(nS, nA))
        pi_init = pi_init / np.sum(pi_init, axis=1)[:, np.newaxis]
        mix_Ps = generate_mix_Ps_Orth(n=ntrain + ntest, nS=nS, nA=nA, eps_lst=eps_lst, seed=seed)

        for eps_index in range(eps_lst_len):
            Ps = mix_Ps[eps_index]

            center_P = [Ps[0]]

            train_P = Ps[:ntrain]
            test_P = Ps[ntrain:]

            MEObj_lst = np.zeros((E_size, args.iter))
            test_MEObj_lst = np.zeros((E_size, args.iter))

            test_center_MEObj_lst = np.zeros((E_size, args.iter))

            MEObj_base_lst = np.zeros((args.iter))  # Baseline
            MEObj_base_avg_lst = np.zeros((args.iter))  # Baseline2
            test_MEObj_base_lst = np.zeros((args.iter))  # Baseline
            test_MEObj_base_avg_lst = np.zeros((args.iter))  # Baseline2

            # ProjPAvg with different Es
            for E_num in range(E_size-1):
                E = E_lst[E_num]
                pi = pi_init.copy()
                lr = args.lr
                for e in range(args.iter//E):
                    # If the decrease is too small or negative, reduce the lr
                    if e > 2 and (MEObj_lst[E_num][(e - 1)*E] - MEObj_lst[E_num][(e - 2)*E]) < MEObj_lst[E_num][(e - 2)*E] * 1e-3:
                        lr = lr * args.lr_decay

                    center_ME_Obj = evaluate_MEObj_from_policy(pi, R, center_P, d_0, gamma)

                    ME_Obj = evaluate_MEObj_from_policy(pi, R, train_P, d_0, gamma)
                    ME_Obj_test = evaluate_MEObj_from_policy(pi, R, test_P, d_0, gamma)
                    for ttt in range(E):
                        MEObj_lst[E_num][e * E + ttt] = ME_Obj
                        test_MEObj_lst[E_num][e * E + ttt] = ME_Obj_test

                        test_center_MEObj_lst[E_num][e * E + ttt] = center_ME_Obj

                    pis = []
                    for i in range(ntrain):
                        local_pi = pi.copy()
                        for _ in range(E):
                            grad = project_policy_gradient(local_pi, train_P[i], R, d_0, gamma)
                            local_pi += lr * grad
                            local_pi = mat2simplex(local_pi.T).T
                        pis.append(local_pi)
                    pi = sum(pis) / ntrain

            # Baseline: Train every agent separately, i.e. do not merge
            lr = args.lr
            pi = pi_init.copy()
            pis = [pi.copy() for _ in range(ntrain)]
            avg_pi = pi.copy()
            for e in range(args.iter):
                if e > 2 and (MEObj_lst[-1][e - 1] - MEObj_lst[-1][e - 2]) < MEObj_lst[-1][e - 2] * 1e-3:
                    lr = lr * args.lr_decay
                ME_Obj = evaluate_MEObj_from_policy(avg_pi, R, train_P, d_0, gamma)
                ME_Obj_test = evaluate_MEObj_from_policy(avg_pi, R, test_P, d_0, gamma)

                center_ME_Obj = evaluate_MEObj_from_policy(avg_pi, R, center_P, d_0, gamma)

                MEObj_lst[-1][e] = ME_Obj
                test_MEObj_lst[-1][e] = ME_Obj_test

                test_center_MEObj_lst[-1][e] = center_ME_Obj

                for i in range(ntrain):
                    local_pi = pis[i]

                    grad = project_policy_gradient(local_pi, train_P[i], R, d_0, gamma)
                    local_pi += lr * grad
                    local_pi = mat2simplex(local_pi.T).T

                    pis[i] = local_pi
                avg_pi = sum(pis) / ntrain

                best_ME_Obj = -inf
                best_label = -1
                avg_ME_Obj = 0
                avg_ME_Obj_test = 0
                ME_Obj_tests = []
                for i in range(ntrain):
                    i_pi = pis[i]
                    i_ME_Obj = evaluate_MEObj_from_policy(i_pi, R, train_P, d_0, gamma)
                    avg_ME_Obj += i_ME_Obj
                    if i_ME_Obj > best_ME_Obj:
                        best_ME_Obj = i_ME_Obj
                        best_label = i
                    # Test
                    i_ME_Obj_test = evaluate_MEObj_from_policy(i_pi, R, test_P, d_0, gamma)
                    avg_ME_Obj_test += i_ME_Obj_test
                    ME_Obj_tests.append(i_ME_Obj_test)
                MEObj_base_lst[e] = best_ME_Obj
                MEObj_base_avg_lst[e] = avg_ME_Obj / ntrain
                test_MEObj_base_avg_lst[e] = avg_ME_Obj_test / ntrain
                test_MEObj_base_lst[e] = ME_Obj_tests[best_label]

            MEObj_lst = MEObj_lst - 0.5 / (1 - gamma)
            test_MEObj_lst = test_MEObj_lst - 0.5 / (1 - gamma)
            MEObj_base_avg_lst = MEObj_base_avg_lst - 0.5 / (1 - gamma)
            test_MEObj_base_avg_lst = test_MEObj_base_avg_lst - 0.5 / (1 - gamma)
            MEObj_base_lst = MEObj_base_lst - 0.5 / (1 - gamma)
            test_MEObj_base_lst = test_MEObj_base_lst - 0.5 / (1 - gamma)

            test_center_MEObj_lst = test_center_MEObj_lst - 0.5 / (1 - gamma)

            total_MEObj_lst[count][eps_index] = MEObj_lst
            total_MEObj_base_lst[count][eps_index] = MEObj_base_lst
            total_MEObj_base_avg_lst[count][eps_index] = MEObj_base_avg_lst
            total_test_MEObj_lst[count][eps_index] = test_MEObj_lst
            total_test_MEObj_base_lst[count][eps_index] = test_MEObj_base_lst
            total_test_MEObj_base_avg_lst[count][eps_index] = test_MEObj_base_avg_lst

            total_test_center_MEObj_lst[count][eps_index] = test_center_MEObj_lst

    np.save(save_dir + '/MEOBj_lst.npy', total_MEObj_lst)
    np.save(save_dir + '/MEOBj_base_lst.npy', total_MEObj_base_lst)
    np.save(save_dir + '/MEOBj_base_avg_lst.npy', total_MEObj_base_avg_lst)

    np.save(save_dir + '/test_MEOBj_lst.npy', total_test_MEObj_lst)
    np.save(save_dir + '/test_MEOBj_base_lst.npy', total_test_MEObj_base_lst)
    np.save(save_dir + '/test_MEOBj_base_avg_lst.npy', total_test_MEObj_base_avg_lst)

    np.save(save_dir + '/test_center_MEOBj_lst.npy', total_test_center_MEObj_lst)
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
