import numpy as np
from numpy.linalg import norm, svd
import cvxpy as cp
import math


def clac_coherence(F):
    H = np.dot(F.T, F)
    for i in range(0, len(F.T)):
        H[(i, i)] = 0
    coherence = norm(H, ord=np.Inf)
    return coherence


def l2_normalize_col(A):
    return A / norm(A, ord=None, axis=0)


def solve_cvxopt(cb_m, cb_N, H, T_i, i):
    n = cb_m + 1

    f = np.zeros(n)
    f[cb_m] = 1
    A = np.diag(np.append(np.ones(cb_m), np.zeros(1)))
    b = np.append(H.T[i], np.zeros(1))
    c = np.zeros(n)
    d = math.sqrt(T_i)
    F = np.append(np.delete(H, i, axis=1), -np.ones(cb_N-1).reshape(1, cb_N-1), axis=0).T
    g = np.zeros(cb_N - 1)

    # Define and solve the CVXPY problem.
    x = cp.Variable(n)
    # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
    soc_constraints = [
        cp.SOC(c.T @ x + d, A @ x - b)
    ]
    prob = cp.Problem(cp.Minimize(f.T @ x),
                      soc_constraints + [F @ x <= g])
    prob.solve()
    return x.value[:cb_m]


def real_valued_frame(m, N, K):
    # Initialization
    H = np.random.randn(m, N)   # m*N dim
    H_normed = l2_normalize_col(H)
    U, Sigma, VT = svd(H_normed, full_matrices=False)
    H = np.dot(U, VT)
    H_normed = l2_normalize_col(H)
    last_coherence = clac_coherence(H_normed)
    print("Original H = \n", H)
    print("Original H's Coherence is ", last_coherence)
    # Iterations
    cnt = 1
    while cnt <= K:
        cnt += 1
        G = np.dot(H_normed.T, H_normed)
        G -= np.diag(np.diag(G))    # N*N dim

        T = 1 - norm(G, ord=np.Inf, axis=0) ** 2
        T *= np.random.random(N)
        for i in range(0, N):
            h_i = solve_cvxopt(m, N, H_normed, T[i], i)
            h_i /= norm(h_i, ord=None)
            H.T[i] = h_i

        if clac_coherence(H) < last_coherence:
            U, Sigma, VT = svd(H_normed, full_matrices=False)
            H = np.dot(U, VT)
            H_normed = l2_normalize_col(H)
    return H_normed


if __name__ == '__main__':
    m = 3
    N = 5
    K = 5000
    Welsbond = math.sqrt((N - m) / ((N - 1) * m))
    H = real_valued_frame(m, N, K)
    miu = clac_coherence(H)
    print("Final H = \n", H)
    print("Final H's Coherence is ", miu)
    print("Welsbond is ", Welsbond)