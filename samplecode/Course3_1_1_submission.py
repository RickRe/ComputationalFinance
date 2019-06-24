# encoding: utf-8

import numpy as np
import model
import utils
import plot_data


def exec_c3_1_a(X_a, X_b, init_w):
    """
    plot 3 histogram of data projecting to difference vector w
    :param X_a: Gaussian data of class a
    :param X_b: Gaussian data of class b
    :param init_w: initial w vector to be projected
    :return: none
    """
    n_histogram = 3
    proj_a = np.zeros((X_a.shape[0], n_histogram))
    proj_b = np.zeros((X_b.shape[0], n_histogram))
    new_w = np.zeros((init_w.shape[0], n_histogram))

    for i in range(n_histogram):
        new_w[:, i] = (init_w + np.array(np.random.randn(*init_w.shape))).ravel()
        proj_a[:, i] = utils.project_X_onto_w(X_a, new_w[:, i]).ravel()
        proj_b[:, i] = utils.project_X_onto_w(X_b, new_w[:, i]).ravel()

    plot_data.plt_histogram(proj_a, proj_b, new_w)


def exec_c3_1_b(X_a, X_b, init_w):
    """
    Turn vector w by 360 degree to find the maximum value of Fisher score,
    and the corresponding direction w∗
    :param X_a: Gaussian data of class a
    :param X_b: Gaussian data of class b
    :param init_w: initial w vector to be projected
    :return: none
    """
    fs_clf = model.FisherScoreClassifier(X_a, X_b, init_w)
    optimal_w = fs_clf.classify()


if __name__ == '__main__':
    # generate gaussian distribution for class a
    n_pts = 100
    mean_a = [4, 2]
    cov_a = np.array([[1, 0.5], [0.5, 1]])  # diagonal covariance
    Gaus_dist_a = model.GausDS(mean_a, cov_a, n_pts)

    # generate gaussian distribution for class b
    mean_b = [2, 4]
    cov_b = np.array([[1, 0.5], [0.5, 1]])  # diagonal covariance
    Gaus_dist_b = model.GausDS(mean_b, cov_b, n_pts)

    # plot two Gaussian distributions including class a and class b
    plot_data.plt_distribution(Gaus_dist_a.data, Gaus_dist_b.data)

    # init weight to do projection
    init_w = np.array([1, -2]).reshape(-1, 1)

    # draw three histograms by projecting to different w
    exec_c3_1_a(Gaus_dist_a.data, Gaus_dist_b.data, init_w)

    # find optimal angel to separate class a and class b
    exec_c3_1_b(Gaus_dist_a.data, Gaus_dist_b.data, init_w)

