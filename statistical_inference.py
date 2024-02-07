import numpy as np
from typing import Callable
from scipy.stats import norm
from scipy.stats import kendalltau


def holm_step_down(p_value: np.ndarray, a: float) -> np.ndarray:
    """
    Holm Step Down procedure for testing M=N(N-1)/2 hypotheses
    with tests of the form:
    1(the hypothesis is rejected) if p_value[i,j]<a_ij,
    0(the hypothesis is accepted) if p_value[i,j]>=a_ij.

    It is known that for this procedure FWER<=a.

    Parameters
    ----------
    p_value : (N,N) ndarray
        Matrix of p-values.

    a : float
        The boundary of FWER.

    Returns
    -------
    decision_matrix : (N,N) ndarray
        Decision matrix.

    """
    N = p_value.shape[0]
    M = N * (N - 1) // 2
    decision_matrix = np.zeros((N, N), dtype=int)
    p_value_array = []
    for i in range(N):
        for j in range(i + 1, N):
            p_value_array.append((p_value[i][j], i, j))
    p_value_array.sort()
    for k in range(M):
        if p_value_array[k][0] >= a / (M - k):
            break
        else:
            decision_matrix[p_value_array[k][1]][p_value_array[k][2]] = 1
            decision_matrix[p_value_array[k][2]][p_value_array[k][1]] = 1
    return decision_matrix


def _corr_calculation(
    x: np.ndarray, transformer: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    N, n = x.shape
    corr = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            corr[i][j] = np.sum(transformer(x[i] * x[j])) / n
            corr[j][i] = corr[i][j]
        corr[i][i] = 1
    return corr


def sign_similarity(x: np.ndarray) -> np.ndarray:
    """
    Calculates a sample sign similarity matrix.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional random vector.

    Returns
    -------
    corr : (N,N) ndarray
        Sample sign similarity matrix.

    """
    x = np.array(x).T
    N, n = x.shape
    mean = np.mean(x, axis=1).reshape((N, -1))
    x = x - mean
    transformer = np.vectorize(lambda y: 1 if y >= 0 else 0)
    return _corr_calculation(x, transformer)


def sign_similarity_statistics(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Calculates a matrix of statistics
    where each statistic has a standard Gaussian distribution N(0,1)
    under the assumption that the sign measure of similarity
    between the i and j component of the elliptical random vector
    is equal to the threshold.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional elliptical random vector.

    threshold : float
        The threshold in the interval (0, 1).

    Returns
    -------
    statistics : (N,N) ndarray
        Matrix of statistics.

    """
    n, N = x.shape
    return np.sqrt(n) * (sign_similarity(x) - threshold) / np.sqrt(threshold * (1 - threshold))


def threshold_graph_p_value(statistics: np.ndarray) -> np.ndarray:
    p_value = np.vectorize(lambda y: 1 - norm.cdf(y))
    return p_value(statistics)


def concentration_graph_p_value(statistics: np.ndarray) -> np.ndarray:
    p_value = np.vectorize(lambda y: 2 * (1 - norm.cdf(np.abs(y))))
    return p_value(statistics)


def from_pearson_to_sign(pearson):
    return 0.5 + (1 / np.pi) * np.arcsin(pearson)


def partial_pearson_statistics(X: np.ndarray) -> np.ndarray:
    X = np.array(X)
    n, N = X.shape
    PEARSON = [[0 for i in range(N)] for j in range(N)]
    cov = np.array(np.cov(X.T, ddof=n-1))
    cov_inv = np.linalg.inv(cov)
    for i in range(N):
        for j in range(i+1, N):
            PEARSON[i][j] = -cov_inv[i][j] / np.sqrt(cov_inv[i][i] * cov_inv[j][j])
            PEARSON[i][j] = np.sqrt(n-N-1) * np.arctanh(PEARSON[i][j])
            PEARSON[j][i] = PEARSON[i][j]
    return np.array(PEARSON)


def kendall_residual_statistics(X: np.ndarray) -> np.ndarray:
    X = np.array(X)
    n, N = X.shape
    KENDALL = [[0 for i in range(N)] for j in range(N)]
    A = np.matrix(np.cov(X.T, ddof=n-1))
    for i in range(N):
        for j in range(i+1, N):
            mask = [it for it in range(N) if it != i and it != j]
            A_12 = A[[i, j], :][:, mask]
            A_22 = A[mask, :][:, mask]
            beta = np.dot(A_12, np.linalg.inv(A_22))
            Xij = X[:, mask]
            Xi = X[:, i]
            Xj = X[:, j]
            beta_i = np.array(beta[0, :])[0]
            beta_j = np.array(beta[1, :])[0]
            res_i = Xi - np.dot(Xij, beta_i)
            res_j = Xj - np.dot(Xij, beta_j)
            KENDALL[i][j] = np.sqrt((9*(n-N+2)*(n-N+1))/(2*(2*n-2*N+9))) * kendalltau(res_i, res_j)[0]
            KENDALL[j][i] = KENDALL[i][j]
    return np.array(KENDALL)
