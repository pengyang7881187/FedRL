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
parser.add_argument("--lr", default=5e-2, type=float)
parser.add_argument("--lr_decay", default=0.8, type=float)
parser.add_argument("--n", default=5, type=int)
parser.add_argument("--eps", default=0.8, type=float)
parser.add_argument("--sparsity", default=0.2, type=float)
parser.add_argument("--gen", default="random", type=str)
parser.add_argument("--iter", default=2048)
args = parser.parse_args()

nS, nA, gamma = args.nS, args.nA, args.gamma
gen, eps = args.gen, args.eps
ntrain = args.n
ntest = 4 * ntrain
sparsity = args.sparsity

d_0 = np.ones(nS)/nS

outer_iter_num = 16000

Seed = 100

E_lst = (1, 2, 4, 8, 16, 32)

E_size = len(E_lst) + 1

separation_len = 50

save_dir = './plot-SoftPAvg-random_MDP'


def experiment():
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.random.seed(Seed)
    seeds = np.random.randint(low=0, high=1000000, size=outer_iter_num)

    total_MEObj_lst = np.zeros((outer_iter_num, E_size, args.iter))
    total_test_MEObj_lst = np.zeros((outer_iter_num, E_size, args.iter))

    total_MEObj_base_lst = np.zeros((outer_iter_num, args.iter))  # Baseline
    total_MEObj_base_avg_lst = np.zeros((outer_iter_num, args.iter))  # Baseline2
    total_test_MEObj_base_lst = np.zeros((outer_iter_num, args.iter))  # Baseline
    total_test_MEObj_base_avg_lst = np.zeros((outer_iter_num, args.iter))  # Baseline2

    for count in tqdm(range(outer_iter_num)):
        # Generate environments
        seed = seeds[count]
        R = np.random.uniform(low=0, high=1, size=(nS, nA))
        pi_logit_init = np.random.uniform(size=(nS, nA))
        Ps = generate_Ps(ntrain + ntest, nS, nA, gen=gen, eps=eps, sparsity=sparsity, seed=seed)
        train_P = Ps[:ntrain]
        test_P = Ps[ntrain:]

        MEObj_lst = np.zeros((E_size, args.iter))
        test_MEObj_lst = np.zeros((E_size, args.iter))
        MEObj_base_lst = np.zeros((args.iter))  # Baseline
        MEObj_base_avg_lst = np.zeros((args.iter))  # Baseline2
        test_MEObj_base_lst = np.zeros((args.iter))  # Baseline
        test_MEObj_base_avg_lst = np.zeros((args.iter))  # Baseline2

        # SoftPAvg with different Es
        for E_num in range(E_size-1):
            E = E_lst[E_num]
            pi_logit = pi_logit_init.copy()
            lr = args.lr
            for e in range(args.iter//E):
                # If the decrease is too small or negative, reduce the lr
                if e > 2 and (MEObj_lst[E_num][(e - 1)*E] - MEObj_lst[E_num][(e - 2)*E]) < MEObj_lst[E_num][(e - 2)*E] * 1e-3:
                    lr = lr * args.lr_decay
                pi = np_softmax(pi_logit)
                ME_Obj = evaluate_MEObj_from_policy(pi, R, train_P, d_0, gamma)
                ME_Obj_test = evaluate_MEObj_from_policy(pi, R, test_P, d_0, gamma)
                for ttt in range(E):
                    MEObj_lst[E_num][e * E + ttt] = ME_Obj
                    test_MEObj_lst[E_num][e * E + ttt] = ME_Obj_test

                pis_logit = []
                for i in range(ntrain):
                    local_pi_logit = pi_logit.copy()
                    for _ in range(E):
                        grad = softmax_policy_gradient(local_pi_logit, train_P[i], R, gamma)
                        local_pi_logit += lr * grad
                    pis_logit.append(local_pi_logit)
                pi_logit = sum(pis_logit) / ntrain

        # Baseline: Train every agent separately, i.e. do not merge
        lr = args.lr
        pi_logit = pi_logit_init.copy()
        pis_logit = [pi_logit.copy() for _ in range(ntrain)]
        avg_pi_logit = pi_logit.copy()
        for e in range(args.iter):
            if e > 2 and (MEObj_lst[-1][e - 1] - MEObj_lst[-1][e - 2]) < MEObj_lst[-1][e - 2] * 1e-3:
                lr = lr * args.lr_decay
            avg_pi = np_softmax(avg_pi_logit)
            ME_Obj = evaluate_MEObj_from_policy(avg_pi, R, train_P, d_0, gamma)
            ME_Obj_test = evaluate_MEObj_from_policy(avg_pi, R, test_P, d_0, gamma)

            MEObj_lst[-1][e] = ME_Obj
            test_MEObj_lst[-1][e] = ME_Obj_test

            for i in range(ntrain):
                local_pi_logit = pis_logit[i]

                grad = softmax_policy_gradient(local_pi_logit, train_P[i], R, gamma)
                local_pi_logit += lr * grad

                pis_logit[i] = local_pi_logit
            avg_pi_logit = sum(pis_logit) / ntrain

            best_ME_Obj = -inf
            best_label = -1
            avg_ME_Obj = 0
            avg_ME_Obj_test = 0
            ME_Obj_tests = []
            for i in range(ntrain):
                i_pi_logit = pis_logit[i]
                i_pi = np_softmax(i_pi_logit)
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

        total_MEObj_lst[count] = MEObj_lst
        total_MEObj_base_lst[count] = MEObj_base_lst
        total_MEObj_base_avg_lst[count] = MEObj_base_avg_lst
        total_test_MEObj_lst[count] = test_MEObj_lst
        total_test_MEObj_base_lst[count] = test_MEObj_base_lst
        total_test_MEObj_base_avg_lst[count] = test_MEObj_base_avg_lst

    np.save(save_dir + '/MEOBj_lst.npy', total_MEObj_lst)
    np.save(save_dir + '/MEOBj_base_lst.npy', total_MEObj_base_lst)
    np.save(save_dir + '/MEOBj_base_avg_lst.npy', total_MEObj_base_avg_lst)

    np.save(save_dir + '/test_MEOBj_lst.npy', total_test_MEObj_lst)
    np.save(save_dir + '/test_MEOBj_base_lst.npy', total_test_MEObj_base_lst)
    np.save(save_dir + '/test_MEOBj_base_avg_lst.npy', total_test_MEObj_base_avg_lst)
    return


def report():
    print('Save dir=' + save_dir)
    coff = np.sqrt(outer_iter_num)
    total_MEOBj_lst = np.load(save_dir + '/MEOBj_lst.npy')
    total_test_MEOBj_lst = np.load(save_dir + '/test_MEOBj_lst.npy')

    for E_num in range(E_size):
        print('-' * separation_len)
        if E_num == E_size - 1:
            print('E=Inf:')
        else:
            print(f'E={E_lst[E_num]}:')
        train_mean = np.max(np.average(total_MEOBj_lst[:, E_num, :], axis=0))
        train_std = np.std(total_MEOBj_lst[:, E_num, :], axis=0)[-1] / coff
        test_mean = np.max(np.average(total_test_MEOBj_lst[:, E_num, :], axis=0))
        test_std = np.std(total_test_MEOBj_lst[:, E_num, :], axis=0)[-1] / coff
        print(f'Train: Mean={train_mean}, Std={train_std}')
        print(f'Test: Mean={test_mean}, Std={test_std}')
    return


if __name__ == '__main__':
    experiment()
    report()
