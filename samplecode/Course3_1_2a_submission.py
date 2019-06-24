# encoding: utf-8

import numpy as np
import model
import plot_data


if __name__ == '__main__':
    # generate gaussian distribution for class a
    n_pts = 100
    mean_a = [4, 0]
    cov_a = np.array([[2, 0], [0, 2]])  # diagonal covariance
    # cov_a = np.array([[1, 0.5], [0.5, 1]])  # diagonal covariance
    Gaus_dist_a = model.GausDS(mean_a, cov_a, n_pts)

    # generate gaussian distribution for class b
    mean_b = [0, 4]
    cov_b = np.array([[2, 0], [0, 2]])  # diagonal covariance
    Gaus_dist_b = model.GausDS(mean_b, cov_b, n_pts)

    # plot probability contours of Gaussian distribution
    plot_data.plot_prob_contours(Gaus_dist_a, Gaus_dist_b)



