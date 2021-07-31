import numpy as np


def get_H_inv(A, b, x):
    d = A.shape[1]
    H = np.zeros((d, d))
    for i in range(A.shape[0]):
        a = A[i:i+1].T
        H += a @ a.T / (a.T @ x - b[i])**2
    try:
        H_inv = np.linalg.inv(H)
    except:
        H_inv = np.linalg.pinv(H)
    return H_inv


def get_sigmas(A, b, x, H_inv):
    sigmas = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        a = A[i:i+1].T
        sigmas[i] = a.T @ H_inv @ a / (a.T @ x - b[i])**2
    return sigmas


def get_Q_inv(sigmas, A, b, x):
    d = A.shape[1]
    Q = np.zeros((d, d))
    for i in range(A.shape[0]):
        a = A[i:i+1].T
        Q += sigmas[i] * a @ a.T / (a.T @ x - b[i])**2
    try:
        Q_inv = np.linalg.inv(Q)
    except:
        Q_inv = np.linalg.pinv(Q)
    return Q_inv


def get_dV(sigmas, A, b, x):
    dV = np.zeros((A.shape[1], 1))
    for i in range(A.shape[0]):
        a = A[i:i+1].T
        dV -= sigmas[i] * a / (a.T @ x - b[i])**2
    return dV


def get_vol_center(A, b, x, n_steps=20, stepsize=0.10):
    vals = np.zeros(n_steps)
    for step in range(n_steps):
        H_inv = get_H_inv(A, b, x)
        vals[step] = 1 / np.linalg.det(H_inv)
        sigmas = get_sigmas(A, b, x, H_inv)
        Q_inv = get_Q_inv(sigmas, A, b, x)
        dV = get_dV(sigmas, A, b, x)
        x = x - stepsize * Q_inv @ dV
    return x


def get_beta(A, c, x, eps, eta, H_inv):
    squared = 2 * c.T @ H_inv @ c / np.sqrt(eps * eta)
    beta = x.T @ c - np.sqrt(squared)
    return beta


def add_row(A, b, c, beta):
    A = np.vstack((A, c.T))
    b = np.append(b, beta)
    return A, b


def remove_row(A, b, i):
    A = np.delete(A, i, 0)
    b = np.delete(b, i)
    return A, b


def vaidya(A_0, b_0, x_0, eps, eta, K, oracle, newton_steps=10, stepsize=0.18, verbose=True):
    """Use Vaidya's method to minimize f(x)."""
    A_k, b_k = A_0, b_0
    x_k = x_0

    xs = [x_0.copy()]
    aux_evals = [0]
    for k in range(K):
        if verbose and k % 20 == 0:
            print(f"k={k}")
        x_k = get_vol_center(A_k, b_k, x_k, n_steps=newton_steps, stepsize=stepsize)
        if (x_k < 0).any():
            break
        H_inv = get_H_inv(A_k, b_k, x_k)
        sigmas = get_sigmas(A_k, b_k, x_k, H_inv)
        if (sigmas >= eps).all():
            c_k = oracle(x_k)
            beta_k = get_beta(A_k, -c_k, x_k, eps, eta, H_inv)
            A_k, b_k = add_row(A_k, b_k, -c_k, beta_k)
        else:
            i = sigmas.argmin()
            A_k, b_k = remove_row(A_k, b_k, i)

        xs.append(x_k.copy())

    return xs


def get_init_polytope(d, R):
    # Задать начальное множество A_0, b_0 для радиуса R
    A_0 = np.vstack((np.eye(d), -np.ones((1, d))))
    b_0 = -R * np.ones(d + 1)
    b_0[-1] *= d
    return A_0, b_0