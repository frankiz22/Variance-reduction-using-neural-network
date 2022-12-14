# This file contains all the payoff functions and other useful functions related to the price generation.

import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt

S0, K, G = 100, 100, 100

sigma = 0.4
rho = 0.75
T = 1

# Number of assets
N = 100

weight = 1 / N

# Interest rate
r = 0.02

Sigma = (rho * np.ones((N,N)) + (1 - rho) * np.eye(N))
L = np.linalg.cholesky(Sigma)

Neuler = 10


# AutoCall
C = [20, 25, 30]
K_AC = 80
T_AC = 3
DB = 40
AB = 100

# Local
Neuler_Local = 100
N_Local = 10
weight_Local = 1 / N_Local

Sigma_Local = (rho * np.ones((N_Local,N_Local)) + (1 - rho) * np.eye(N_Local))
L_Local = np.linalg.cholesky(Sigma_Local)

# Heston
Neuler_Heston = 100
N_Heston = 10
weight_Heston = 1 / N_Heston

kappa = 2
sigma0 = 0.04
a = 0.04
nu = 0.01
gamma = -0.2
rho_Gamma = 0.3

Gamma = (rho_Gamma * np.ones((N_Heston,N_Heston)) + (1 - rho_Gamma) * np.eye(N_Heston))
Gamma_Tilde_high = np.hstack((Gamma, gamma * Gamma))
Gamma_Tilde_low = np.hstack((gamma * Gamma, (gamma**2) * Gamma + (1 - gamma**2) * np.eye(len(Gamma))))
Gamma_Tilde = np.vstack((Gamma_Tilde_high, Gamma_Tilde_low))
L_Tilde = np.linalg.cholesky(Gamma_Tilde)

def S_T_total(L, r, S0, sigma, T, Z, eps=None, idx=0):
    """
    Input:
        L.shape = N, N
        Z.shape = M, N
    Output: output.shape = M, N
    """
    if eps is None:
        return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * (Z @ L.T))
    else:
        output = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * (Z @ L.T))
        output[idx] *= ((S0 + eps) / S0)
        return output
    
def S_t_Path(L, Neuler, r, S0, sigma, T, Z):
    """
    Input: 
        L.shape = N, N
        Z.shape = M, N * Neuler
    Output: 
        output.shape = M, N, Neuler + 1
    """
    dt = T / Neuler
    M = Z.shape[0]
    N = len(L)
    St = np.zeros((M, N, Neuler))
    Z = np.reshape(Z, (M, N, Neuler))
    time = dt * np.arange(1, Neuler + 1)
    for i in range(M):
        B = np.cumsum(L @ Z[i], axis=1)
        St[i] = S0 * np.exp((r - 0.5 * sigma**2) * time + sigma * np.sqrt(dt) * B)
    return np.insert(St, 0, S0, axis=2)

def S_t_Euler(L, Neuler, r, S0, T, Z, a=a, kappa=kappa, model='BS', nu=nu, sigma0=None):
    """
    Input for BS and local volatility model: 
        L.shape = N, N
        Z.shape = M, N * Neuler
    Input for Heston model:
        L.shape = (2*N), (2*N)
        Z.shape = M, (2*N) * Neuler
    Output: 
        output.shape = M, N, Neuler + 1
    """
    dt = T / Neuler
    M = Z.shape[0]

    if model=='BS' or model=='Local':
        N = len(L)
        St = S0 * np.ones((M, N, Neuler + 1))
        Z = np.reshape(Z, (M, N, Neuler))
        if model=='BS':
            sigma = sigma0
            St[:, :, 1:] *= np.cumprod(1 + r * dt + sigma * np.sqrt(dt) * (L @ Z[:, :, :].T).T, axis=2)
        else:
            t = 0
            for j in range(Neuler):
                sigma = local_volatility(r, S0, t, St[:, :, j])
                St[:, :, j + 1] = St[:, :, j] * (1 + r * dt + sigma * np.sqrt(dt) * (L @ Z[:, :, j].T).T)
                t += dt
    elif model=='Heston':
        N = len(L) // 2
        St = S0 * np.ones((M, N, Neuler + 1))
        Z = np.reshape(Z, (M, 2*N, Neuler))
        sigma = sigma0
        for j in range(Neuler):
            St[:, :, j + 1] = St[:, :, j] * (1 + r * dt + np.sqrt(sigma * dt) * ((L @ Z[:, :, j].T).T)[:,:N])
            sigma += kappa * (a - sigma) * dt + nu * np.sqrt(sigma * dt) * ((L @ Z[:, :, j].T).T)[:,N:]
    return St

def local_volatility(r, s, t, x):
    return 0.6 * (1.2 - np.exp(-0.1 * t - 0.001 * (x * np.exp(r * t) - s)**2)) * np.exp(-0.05 * np.sqrt(t))


def Call_Basket(K, r, ST, T, weight):
    """
    Input: 
        ST.shape = M, N
        weight.shape = N
    Output: output.shape = M
    """
    return np.exp(-r * T) * np.maximum(0, np.sum(weight * ST, axis=1) - K)

def f_Call_Basket(Z, a=a, K=K, kappa=kappa, L=L, model='BS', Neuler=Neuler, r=r, S0=S0, sigma=sigma, T=T, weight=weight, is_Euler=False):
    """
    Input if is_Euler:
        Z.shape = M, (N*2) * Neuler or M, N * Neuler (depending on the model)
    Input else:
        Z.shape = M, N
    Output: 
        output.shape = M
    """
    if is_Euler:
        St = S_t_Euler(L, Neuler, r, S0, T, Z, a=a, kappa=kappa, model=model, sigma0=sigma)
        ST = St[:,:,-1]
    else:
        ST = S_T_total(L, r, S0, sigma, T, Z)
    return Call_Basket(K, r, ST, T, weight)

def Put_Worst_Of(K, r, ST, T):
    return np.exp(-r * T) * np.maximum(0, K - np.min(ST, axis=1))

def f_Put_Worst_Of(Z, a=a, K=K, kappa=kappa, L=L, model='BS', Neuler=Neuler, r=r, S0=S0, sigma=sigma, T=T, is_Euler=False):
    if is_Euler:
        St = S_t_Euler(L, Neuler, r, S0, T, Z, a=a, kappa=kappa, model=model, sigma0=sigma)
        ST = St[:,:,-1]
    else:
        ST = S_T_total(L, r, S0, sigma, T, Z)
    return Put_Worst_Of(K, r, ST, T)

def Digit_Basket(G, K, r, ST, T, weight):
    return np.exp(-r * T) * G * (np.sum(weight * ST, axis=1) >= K)

def f_Digit_Basket(Z, a=a, G=G, K=K, kappa=kappa, L=L, model='BS', Neuler=Neuler, r=r, S0=S0, sigma=sigma, T=T, weight=weight, is_Euler=False):
    if is_Euler:
        St = S_t_Euler(L, Neuler, r, S0, T, Z, a=a, kappa=kappa, model=model, sigma0=sigma)
        ST = St[:,:,-1]
    else:
        ST = S_T_total(L, r, S0, sigma, T, Z)
    return Digit_Basket(G, K, r, ST, T, weight)


def Asian_Arithmetic(K, r, St, T, weight):
    """
    Input: 
        St.shape = M, N, Neuler + 1
    Output:
        output.shape = M
    """
    Neuler = np.shape(St)[2] - 1
    return np.exp(-r * T) * np.maximum(0, (1 / Neuler) * np.sum(np.sum(weight * St[:,:,1:], axis=1), axis=1) - K)

def f_Asian_Arithmetic(Z, a=a, K=K, kappa=kappa, L=L, model='BS', Neuler=Neuler, r=r, S0=S0, sigma=sigma, T=T, weight=weight, is_Euler=False):
    """
    Input:
        Z.shape = M, N * Neuler
    Output:
        output.shape = M
    """
    if not(is_Euler) and model=='BS':
        St = S_t_Path(L, Neuler, r, S0, sigma, T, Z)
    else:
        St = S_t_Euler(L, Neuler, r, S0, T, Z, a=a, kappa=kappa, model=model, nu=nu, sigma0=sigma)
    return Asian_Arithmetic(K, r, St, T, weight)

def AutoCall_One_Option(AB, C, DB, K, r, St, T, weight):
    """
    AutoCall with a basket performance
    Input:
        St.shape = N, Neuler + 1
    Output:
        output.shape = 1
    """
    Neuler = St.shape[1] - 1
    for i in range(len(C)):
        perf = np.sum(St[:, int((i + 1) * (Neuler + 1) / len(C)) - 1] * weight)
        if perf >= AB:
            return C[i] * np.exp(-r *T)
        elif i==len(C)-1:
            return - max(0, K - perf) * (perf < DB) * np.exp(-r * T)
    
def f_AutoCall(Z, AB=AB, C=C, DB=DB, K=K_AC, L=L, Neuler=Neuler, r=r, S0=S0, sigma=sigma, T=T_AC, weight=weight):
    """
    Input:
        Z.shape = M, N * Neuler
    Output:
        output.shape = M
    """    
    M = Z.shape[0]
    output = np.zeros(M)
    St = S_t_Path(L, Neuler, r, S0, sigma, T, Z)
    for i in range(M):
        output[i] = AutoCall_One_Option(AB, C, DB, K, r, St[i], T, weight)
    return output

def Analytic_EHZ(b1, b2, W1, W2):
    """
    Input:
        b1.shape = n
        b2.shape = 1
        W1.shape = n, N
        W2.shape = 1, n
    Output: 
        output.shape = 1
    """
    n = W1.shape[0]
    inter = np.zeros(n)
    for i in range(n):
        mu_i = b1[i]
        sigma_i = np.sqrt(np.sum(W1[i] * W1[i]))
        inter[i] = sigma_i * np.exp(-0.5 * (mu_i / sigma_i)**2) / np.sqrt(2 * np.pi) + mu_i * (1-sps.norm.cdf(-mu_i / sigma_i))
    return W2 @ inter + b2

def Delta_Basket(K, r, S0, ST, T, weight):
    """
    Input:
        ST.shape = M, N
    Output:
        output.shape = 1
    """
    # We only consider the first asset
    idx = 0
    return np.exp(-r * T) * weight * np.mean(ST[:, idx] * (np.sum(weight * ST, axis=1) >= K) / S0)

def Vega_Basket(K, L, r, sigma, ST, T, weight, Z):
    idx = 0
    return np.exp(-r * T) * weight * np.mean(ST[:, idx] * (np.sum(weight * ST, axis=1) >= K) * ((Z @ L.T)[:, idx] - sigma * T))

def Delta_Basket_2(eps, K, L, r, S0, sigma, T, weight, Z):
    STp = S_T_total(L, r, S0, sigma, T, Z, eps=eps)
    STm = S_T_total(L, r, S0, sigma, T, Z, eps=-eps)
    return np.mean(Call_Basket(K, r, STp, T, weight) - Call_Basket(K, r, STm, T, weight)) / (2 * eps)