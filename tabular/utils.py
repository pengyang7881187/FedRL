import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from numpy.linalg import solve
from GridWorldEnvironment import WindyCliffGridWorld

inf = np.inf


# Occupancy measure
def d_pi(gamma, P_pi, d0):
    nS, _ = P_pi.shape
    dpi = solve((np.eye(nS) - gamma * P_pi).T, d0) * (1 - gamma)
    return dpi


def QtoV(Q):
    return np.max(Q, axis=1)


def VtoQ(V, P, R, gamma):
    nS, nA = R.shape
    return R + gamma * np.inner(P.reshape(nS, nA, nS), V)


def QtoPolicy(Q):
    nS, nA = Q.shape
    pi = np.zeros((nS, nA))
    pi[np.arange(nS), np.argmax(Q, axis=1)] = 1
    return pi


def evaluate_MEObj_from_policy(pi, R, Ps, d_0, gamma):
    nS, nA = R.shape
    R_pi = np.sum(pi * R, axis=1)
    pi_axis = pi[:, :, np.newaxis]
    Ps_pi = [np.sum(P.reshape(nS, nA, nS) * pi_axis, axis=1) for P in Ps]
    Vs = [solve(np.eye(nS) - gamma * P_pi, R_pi) for P_pi in Ps_pi]
    avg_V = np.array([np.inner(V, d_0) for V in Vs])
    ME_Obj = np.mean(avg_V)
    return ME_Obj


def value_iteration(V, P, R, gamma):
    nS, nA = R.shape
    Q = VtoQ(V, P, R, gamma)
    # Return the new V and the new deterministic policy
    new_V = np.max(Q, axis=1)
    new_pi_idx = np.argmax(Q, axis=1)
    new_pi = (np.arange(nA) == new_pi_idx[:, np.newaxis]).astype(float)
    return new_V, new_pi


def np_softmax(logit):
    softmax = nn.Softmax(dim=logit.ndim - 1)
    return softmax(torch.from_numpy(logit)).numpy()


def softmax_policy_gradient(pi_logit, P, R, gamma):
    nS, nA = R.shape
    pi = np_softmax(pi_logit)
    R_pi = np.sum(pi * R, axis=1)
    pi_axis = pi[:, :, np.newaxis]
    P_pi = np.sum(P.reshape(nS, nA, nS) * pi_axis, axis=1)
    V_pi = solve(np.eye(nS) - gamma * P_pi, R_pi)
    Q_pi = VtoQ(V_pi, P, R, gamma)
    A_pi = Q_pi - V_pi[:, np.newaxis]
    grad = A_pi / (1 - gamma)
    return grad


def project_policy_gradient(pi, P, R, d_0, gamma):
    nS, nA = R.shape
    R_pi = np.sum(pi * R, axis=1)
    pi_axis = pi[:, :, np.newaxis]
    P_pi = np.sum(P.reshape(nS, nA, nS) * pi_axis, axis=1)
    V_pi = solve(np.eye(nS) - gamma * P_pi, R_pi)
    Q_pi = VtoQ(V_pi, P, R, gamma)
    dpi = d_pi(gamma, P_pi, d_0)
    grad = dpi[:, np.newaxis] * Q_pi.reshape(nS, nA) / (1 - gamma)
    return grad


def mat2simplex(matX, l=1.):
    m, n = matX.shape
    matS = np.sort(matX, axis=0)[::-1]
    matC = np.cumsum(matS, axis=0) - l
    matH = matS - matC / (np.arange(m) + 1).reshape(m, 1)
    matH[matH <= 0] = np.inf
    r = np.argmin(matH, axis=0)
    t = matC[r, np.arange(n)] / (r + 1)
    matY = matX - t
    matY[matY < 0] = 0
    return matY


# epsilon represent the scale of perturbation
def generate_mix_Ps(n, nS, nA, eps_lst=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), seed=250):
    np.random.seed(seed)
    eps_lst_len = len(eps_lst)
    mix_Ps = []
    center_P = np.random.uniform(0, 1, size=(nA * nS, nS))
    center_P = center_P / np.sum(center_P, 1)[:, np.newaxis]
    noncenter_Ps = []
    for _ in range(n - 1):
        U = np.random.uniform(0, 1, size=(nA * nS, nS))
        U = U / np.sum(U, 1)[:, np.newaxis]
        noncenter_Ps.append(U.copy())
    for i in range(eps_lst_len):
        eps = eps_lst[i]
        current_Ps = []
        current_Ps.append(center_P.copy())
        for j in range(n - 1):
            current_P = center_P * (1 - eps) + eps * noncenter_Ps[j]
            current_Ps.append(current_P.copy())
        mix_Ps.append(current_Ps)
    return mix_Ps


# epsilon represent the scale of perturbation
# Orth
def generate_mix_Ps_Orth(n, nS, nA, eps_lst=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), seed=250):
    np.random.seed(seed)
    eps_lst_len = len(eps_lst)
    mix_Ps = []

    sparsity = 0.5
    sparse = np.random.binomial(1, sparsity, size=(nA * nS, nS))
    for i in range(nA * nS):
        if sum(sparse[i, :]) == 0:
            sparse[i, np.random.randint(nS)] = 1
        if sum(sparse[i, :]) == nS:
            sparse[i, np.random.randint(nS)] = 0

    inverse_sparse = 1 - sparse

    center_P = np.random.uniform(0, 1, size=(nA * nS, nS)) * sparse

    center_P = center_P / np.sum(center_P, 1)[:, np.newaxis]
    noncenter_Ps = []
    for _ in range(n - 1):
        U = np.random.uniform(0, 1, size=(nA * nS, nS)) * inverse_sparse
        U = U / np.sum(U, 1)[:, np.newaxis]
        noncenter_Ps.append(U.copy())
    for i in range(eps_lst_len):
        eps = eps_lst[i]
        current_Ps = []
        current_Ps.append(center_P.copy())
        for j in range(n - 1):
            current_P = center_P * (1 - eps) + eps * noncenter_Ps[j]
            current_Ps.append(current_P.copy())
        mix_Ps.append(current_Ps)
    return mix_Ps


# epsilon represent the scale of perturbation
def generate_mix_windycliff_Ps(n, length_X, length_Y, eps_lst=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), seed=250):
    np.random.seed(seed)
    eps_lst_len = len(eps_lst)
    mix_Ps = []
    center_P = WindyCliffGridWorld(length_X=length_X,
                                   length_Y=length_Y,
                                   gamma=0.9,
                                   theta=0.5).transitions.swapaxes(0, 1)
    noncenter_Ps = []
    for _ in range(n - 1):
        U = WindyCliffGridWorld(length_X=length_X,
                                length_Y=length_Y,
                                gamma=0.9,
                                theta=random.random()).transitions.swapaxes(0, 1)
        noncenter_Ps.append(U.copy())
    for i in range(eps_lst_len):
        eps = eps_lst[i]
        current_Ps = []
        current_Ps.append(center_P.copy())
        for j in range(n - 1):
            current_P = center_P * (1 - eps) + eps * noncenter_Ps[j]
            current_Ps.append(current_P.copy())
        mix_Ps.append(current_Ps)
    return mix_Ps


# epsilon represent the scale of perturbation
def generate_group_windycliff_Ps(length_X, length_Y, para_lst=(0.1, 0.3, 0.5, 0.7, 0.9)):
    group_P = []
    for para in para_lst:
        group_P.append(WindyCliffGridWorld(length_X=length_X,
                                           length_Y=length_Y,
                                           gamma=0.9,
                                           theta=para).transitions.swapaxes(0, 1).copy())
    return group_P


def generate_Ps(n, S, A, gen="random", eps=0.1, seed=250, sparsity=0.05):
    def randomgen(n, nS, nA, seed=250):
        Ps = []
        np.random.seed(seed)
        for _ in range(n):
            sparse = np.random.binomial(1, sparsity, size=(nA * nS, nS))
            for i in range(nA * nS):
                if sum(sparse[i, :]) == 0:
                    sparse[i, np.random.randint(nS)] = 1
            P = sparse * np.random.uniform(0, 1, size=(nA * nS, nS))
            P = P / np.sum(P, 1)[:, np.newaxis]
            Ps.append(P)
        return Ps

    def intergen(n, nS, nA, eps=0.1, seed=250):
        np.random.seed(seed)
        Ps = []
        P = np.random.uniform(0, 1, size=(nA * nS, nS))
        P = P / np.sum(P, 1)[:, np.newaxis]
        for _ in range(n):
            sparse = np.random.binomial(1, 0.05, size=(nA * nS, nS))
            for i in range(nA * nS):
                if sum(sparse[i, :]) == 0:
                    sparse[i, np.random.randint(nS)] = 1
            U = sparse * np.random.uniform(0, 1, size=(nA * nS, nS))
            U = U / np.sum(U, 1)[:, np.newaxis]
            PP = P * (1 - eps) + eps * U
            Ps.append(PP)
        return Ps

    if gen == "random":
        return randomgen(n, S, A, seed)
    else:
        return intergen(n, S, A, eps, seed)
