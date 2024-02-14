from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import pyplot as plt
import transforms3d as t3d
from scipy.spatial.transform import Rotation
import argparse
import copy
from distutils.util import strtobool
from typing import Optional
import numpy as np
np.set_printoptions(threshold=100, linewidth=1000)


def get_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve',
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["umeyama", "malis",],
        default="malis",
        help="LSQ algorithm to use.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=10,
        help="The number of points.",
    )
    parser.add_argument(
        "--num-outliers",
        type=int,
        default=0,
        help="The number of outliers.",
    )

    return parser.parse_args()


def from_r_to_q(r: np.ndarray) -> np.ndarray:
    R = np.reshape(r, (3, 3))
    return Rotation.from_matrix(R).as_quat()


def from_q_to_r(q: np.ndarray) -> np.ndarray:
    r11 = 2 * (q[0]**2 + q[1]**2) - 1
    r12 = 2 * (q[1]*q[2] - q[0]*q[3])
    r13 = 2 * (q[1]*q[3] + q[0]*q[2])
    r21 = 2 * (q[1]*q[2] + q[0]*q[3])
    r22 = 2 * (q[0]**2 + q[2]**2) - 1
    r23 = 2 * (q[2]*q[3] - q[0]*q[1])
    r31 = 2 * (q[1]*q[3] - q[0]*q[2])
    r32 = 2 * (q[2]*q[3] + q[0]*q[1])
    r33 = 2 * (q[0]**2 + q[3]**2) - 1

    return np.asarray([
        r11, r12, r13, r21, r22, r23, r31, r32, r33,
    ])


def printif(*values, flag: bool = True):
    if flag:
        print(*values)


def umeyama(
    X: np.ndarray, Y: np.ndarray, verbose: bool = False,
) -> np.ndarray:
    """A closed-form solution for least square optimization
    via Umeyama algorithm

    Parameters
    ----------
    X : `np.ndarray`
        Array of reference points.
        (d x N) matrix, where d is dimension
        and N is the number of correspondences
    Y : np.ndarray
        Array of target points.
        (d x N) matrix, where d is dimension
        and N is the number of correspondences
    verbose : `bool`
        Whether to print debug messages, by default False

    Returns
    -------
    `np.ndarray`
        _description_
    """
    m, n = X.shape
    printif(m, n, flag=verbose)

    # Mean
    mx = X.mean(axis=1, keepdims=True)
    my = Y.mean(axis=1, keepdims=True)
    Xc = X - mx
    Yc = Y - my

    # Covariance matrix
    Sxy = np.dot(Yc, Xc.T) / n

    U, D, V_t = np.linalg.svd(
        Sxy, full_matrices=True, compute_uv=True,
    )

    r = np.linalg.matrix_rank(Sxy)
    S = np.identity(m)

    if r == m:
        if (np.linalg.det(Sxy) < 0):
            S[m - 1, m - 1] = -1
    elif (r == m - 1):
        if (np.linalg.det(U) * np.linalg.det(V_t) < 0):
            S[m - 1, m - 1] = -1
    else:
        raise ValueError

    # Rotation and translation
    R = U @ S @ V_t
    mx = np.squeeze(mx)
    my = np.squeeze(my)
    t = my - R @ mx

    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def weighted_least_square(
    X: np.ndarray,
    Y: np.ndarray,
    W: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> np.ndarray:
    """A closed-form solution for weighted least square
    with point-to-point correspondences.
    Malis, "Complete Closed-Form and Accurate Solution to Pose Estimation From 3D Correspondences", RA-L, 2023

    Parameters
    ----------
    X : `np.ndarray`
        Array of reference points.
        (d x N) matrix, where d is dimension
        and N is the number of correspondences
    Y : np.ndarray
        Array of target points.
        (d x N) matrix, where d is dimension
        and N is the number of correspondences
        _description_
    W : Optional[np.ndarray], optional
        N dimensional vector of weights, by default None
    verbose : `bool`
        Whether to print debug messages, by default False

    Returns
    -------
    np.ndarray
        4x4 transformation matrix
    """
    m_r_ten = copy.deepcopy(X.T)
    m_k_ten = copy.deepcopy(Y.T)
    printif(m_k_ten, flag=verbose)
    M_k_ten = []
    W_k_ten = []

    if W is None:
        W = np.ones(X.shape[0])

    for k, (m_r, m_k) in enumerate(zip(m_r_ten, m_k_ten)):
        # Weight -> Diag. matrix
        w_k = W[k]
        W_k = np.identity(X.shape[0]) * w_k
        W_k_ten.append(W_k)

        # Point -> Point matrix
        M_k = np.zeros((3, 9))
        M_k[0, 0:3] = m_r
        M_k[1, 3:6] = m_r
        M_k[2, 6:9] = m_r
        M_k_ten.append(M_k)

        # Take matrix sum
        # for (20) - (23)
        WM_k = W_k.T @ M_k
        Wm_k = W_k.T @ m_k

        if k == 0:
            # CAUTION:
            # Initializing by 'W_k_sum = W_k' results in
            # an unexpected behavior.
            W_k_sum = np.array(W_k)
            WM_k_sum = np.array(WM_k)
            Wm_k_sum = np.array(Wm_k)
        else:
            W_k_sum += W_k
            WM_k_sum += WM_k
            Wm_k_sum += Wm_k

    M_k_ten = np.array(M_k_ten)
    W_k_ten = np.array(W_k_ten)

    W_k_sum_inv = np.linalg.inv(W_k_sum)

    A_t = -W_k_sum_inv @ WM_k_sum  # eq. (20)
    b_t = W_k_sum_inv @ Wm_k_sum  # eq. (21)

    # eq. (22), (23)
    M_bar_k_ten = M_k_ten + A_t  # eq. (22)
    m_bar_k_ten = m_k_ten - b_t  # eq. (23)

    # Transpose each matrix in M_bar_k_ten_t
    # Use of '.T' leads to unexpected results
    M_bar_k_ten_t = M_bar_k_ten.transpose([0, 2, 1])
    m_bar_k_ten_ex = np.expand_dims(m_bar_k_ten, axis=-1)
    # m_bar_k_ten_ex_t = m_bar_k_ten_ex.transpose([0, 2, 1])

    # A_r = (M_bar_k_ten_t @ W_k_ten @ M_bar_k_ten).sum(axis=0)
    b_r = -(M_bar_k_ten_t @ W_k_ten @ m_bar_k_ten_ex).sum(axis=0).reshape((-1))
    # c_r = (m_bar_k_ten_ex_t @ W_k_ten @ m_bar_k_ten_ex).sum(axis=0).reshape((-1))

    # The loss function in a form "r.T @ A_r @ r + 2 * b_r.T @ r + c_r"
    # can be rewritten as "2 * b_r.T @ r + const.",
    # which can be further written as a quadratic function of q
    # "q.T @ B @ q + const.".
    # The derivation of B can be found in "loss_function_derivation.ipynb".
    # Refer Malis (2023) for further details.
    B = np.array([
        [2 * (b_r[0]+b_r[4]+b_r[8]), b_r[7] -
         b_r[5], b_r[2]-b_r[6], b_r[3]-b_r[1]],
        [b_r[7]-b_r[5], 2 * b_r[0], b_r[1]+b_r[3], b_r[2]+b_r[6]],
        [b_r[2]-b_r[6], b_r[1]+b_r[3], 2 * b_r[4], b_r[5]+b_r[7]],
        [b_r[3]-b_r[1], b_r[2]+b_r[6], b_r[5]+b_r[7], 2 * b_r[8]],
    ])
    eigval, eigvec = np.linalg.eig(B)
    printif(eigval, flag=verbose)
    printif(eigvec, flag=verbose)

    min_index = np.argmin(eigval)
    min_vec = eigvec[:, min_index]

    r = from_q_to_r(min_vec)

    #
    # Translation
    #
    t = A_t @ r + b_t  # eq. (19)

    T = np.identity(4)
    T[:3, :3] = r.reshape((3, 3))
    T[:3, 3] = t

    return T


if __name__ == "__main__":
    args = get_arguments()

    # Parameters
    dim = 3
    N = args.num_points

    # Randomly generate N points
    np.random.seed(100)
    a = np.random.rand(dim, N,)

    # Randomly initialize a transformation
    rand_trans = np.random.rand(dim)
    rand_rot = Rotation.random().as_matrix()
    T = t3d.affines.compose(
        rand_trans,
        rand_rot,
        np.ones(dim),
    )

    # Homogeneous form
    a_h = np.concatenate(
        [a, np.ones((1, a.shape[-1]))],
        axis=0,
    )

    # Generate transformed points
    b_h = T @ a_h

    # Add gaussian noise
    noise = np.random.standard_normal(
        b_h[:dim, :].shape
    ) / 100
    b = b_h[:dim, :] + noise

    # Set a weight for each correspondences
    W = np.ones(N)

    outlier = True
    if args.num_outliers:
        # Outliers
        if args.num_outliers >= args.num_points:
            print("Too many outliers. I set num_outliers=1")
            num_outliers = 1
        else:
            num_outliers = args.num_outliers

        b[:, -num_outliers:] = np.random.standard_normal((3, num_outliers))

        # Decrease the weight for the outlier
        W[-num_outliers:] /= 100

    if args.algorithm == "malis":
        T_estim = weighted_least_square(a, b, W)
    elif args.algorithm == "umeyama":
        T_estim = umeyama(a, b, )
    else:
        print(f"Algorithm {args.algorithm} is not supported")
        raise ValueError

    b_estim = T_estim @ a_h
    print(f"Per-point error: {np.linalg.norm(b_h - b_estim, axis=0)}")

    print("=== GT transformation ===")
    print(T)
    print("=== Estimated transformation ===")
    print(T_estim)

    # Generate figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # Plot points
    ax.scatter(a[0, :], a[1, :], a[2, :], c='b')
    ax.scatter(b[0, :], b[1, :], b[2, :], c='r')
    ax.scatter(b_estim[0, :], b_estim[1, :], b_estim[2, :], c='y')

    # Show the figure
    plt.show()
