# encoding: utf-8

import numpy as np


def project_X_onto_w(X, v_w):
    """
    project a list of vectors X onto a vector w
    :param X: m-by-2 matrix (m examples with 2 features)
    :param v_w: weights 2-by-1 vector
    :return: projection data of X onto w
    """
    # Rule: projection y(i) = v_x(i) @ v_w / |v_w|
    w_length = np.linalg.norm(v_w)
    assert(w_length > 0)

    # the projection represent the distance between the corresponding projection
    # point and the original point.
    return np.divide((X @ v_w), w_length)


def f_LDA_score(X_a, X_b, init_w, theta, balanced=True):
    """
    Compute Fisher LDA score
    :param X_a: Gaussian data of class a
    :param X_b: Gaussian data of class b
    :param init_w: initial w vector to be projected
    :param theta: the angel which init_w vector to be rotated
    :return: Fisher score
    """
    rotation = [[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]]

    n_a = X_a.shape[0]  # number of examples of class a
    n_b = X_b.shape[0]  # number of examples of class b

    y_a = X_a @ (rotation @ init_w)  # examples of class a project to rotated w
    y_b = X_b @ (rotation @ init_w)  # examples of class b project to rotated w

    u_a = np.mean(y_a, axis=0)
    u_b = np.mean(y_b, axis=0)

    sigma_a_sqr = np.square(y_a - u_a).mean()
    sigma_b_sqr = np.square(y_b - u_b).mean()

    # Fisher score formula
    if balanced:
        fscore = np.divide((u_a - u_b)**2, (n_a/(n_a+n_b))*sigma_a_sqr + (n_b/(n_a+n_b))*sigma_b_sqr)
    else:
        fscore = np.divide((u_a - u_b) ** 2, sigma_a_sqr + sigma_b_sqr)

    return fscore[0]

