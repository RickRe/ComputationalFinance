# encoding: utf-8

import numpy as np
import utils
import plot_data


class GausDS:
    """
    2D Gaussian Data Class. Described by mean and covariance.
    """

    def __init__(self, mean, cov, n_pts):
        self.mean = mean  # mean
        self.cov = cov  # covariance
        self.n_pts = n_pts  # number of data points
        self.data = self.generate_dataset()

    def generate_dataset(self):
        """
        generate Gaussian data distribution
        :return: Gaussian distribution
        """
        return np.random.multivariate_normal(self.mean, self.cov, self.n_pts)


    def Gaussian_pdf(self, xc, yc, ls_num):
        """
        Compute the Gaussian PDF (Probability Density Function)
        :param xc: meshgrid x axis
        :param yc: meshgrid y axis
        :param ls_num: number of points in each axis
        :return: PDF of each point in the meshgrid.
        """
        pos = np.concatenate((np.expand_dims(xc, axis=2), np.expand_dims(yc, axis=2)), axis=2)

        a = (pos - self.mean).dot(np.linalg.inv(self.cov))
        b = np.expand_dims(pos - self.mean, axis=3)
        Z = np.zeros((ls_num, ls_num), dtype=np.float32)

        for i in range(ls_num):
            Z[i] = [np.dot(a[i, j], b[i, j]) for j in range(ls_num)]

        Z = np.exp(Z * (-0.5)) / (((2 * np.pi)**(2/2)) * (np.linalg.det(self.cov)**0.5))
        return Z

        # not working properly
        # pdf = np.zeros([ls_num, ls_num])
        # pos = np.zeros([ls_num, 2])
        # for i in range(ls_num):
        #     pos[:, 0] = xc[:, i]
        #     pos[:, 1] = yc[:, i]
        #     pdf[:, i] = multivariate_normal.pdf(pos, self.mean, self.cov)


class FisherScoreClassifier:
    """
    A binary classifier using Fisher score. Classify two distributions by data projection
    """

    def __init__(self, X_a, X_b, init_w):
        """
        :param X_a: Gaussian data of class a
        :param X_b: Gaussian data of class b
        :param init_w: initial w vector to be projected
        """
        self.X_a = X_a
        self.X_b = X_b
        self.init_w = init_w

    def classify(self, balanced=True, plot=True):
        """
        Turn vector w by 360 degree to find the maximum value of Fisher score,
        and the corresponding direction w∗
        :param balanced: if the fisher score uses the unbalanced formula
        :param plot: plot data if plot is True
        :return: w which provides optimal classification against class a and b
        """
        score_list = []
        max_score = 0
        for theta in np.linspace(-np.pi, np.pi, 360):
            f_score = utils.f_LDA_score(self.X_a, self.X_b, self.init_w, theta, balanced)
            # get the fisher scores
            score_list.append(f_score)
            if f_score > max_score:
                max_score = f_score
                max_theta = theta
                print("max_score %s, max θ %s" % (max_score, max_theta))

        rotation = [[np.cos(max_theta), -np.sin(max_theta)],
                    [np.sin(max_theta), np.cos(max_theta)]]
        w_star = rotation @ self.init_w

        print("The maximum value of F(w(θ)): {}".format(max_score))
        print("The corresponding vector w*: {}".format(w_star))
        print("The corresponding direction θ: %.2f" % max_theta)

        # project X onto w which provides the optimal classification
        y_a = utils.project_X_onto_w(self.X_a, w_star)
        y_b = utils.project_X_onto_w(self.X_b, w_star)

        if plot is True:
            plot_data.plt_optimal_fisher_score(self.X_a, self.X_b, \
                                               score_list, max_theta, \
                                               max_score, w_star, y_a, y_b)
        return w_star
