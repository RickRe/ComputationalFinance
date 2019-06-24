# encoding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import model


def plt_distribution(X_a, X_b):
    """
    Plot two Gaussian distribution
    :param X_a: Gaussian data of class a
    :param X_b: Gaussian data of class b
    :return: none
    """
    plt.scatter(X_a[:, 0], X_a[:, 1], c='r', label='class a')
    plt.scatter(X_b[:, 0], X_b[:, 1], marker='+', c='b', label='class b')
    plt.title("Two 2-dimensional Gaussian distribution")
    plt.xlabel("X1")
    plt.ylabel("X2")
    leg = plt.legend(loc='upper right', fancybox=True, fontsize=8)
    leg.get_frame().set_alpha(0.5)
    plt.show()


def plt_histogram(proj_a, proj_b, w):
    """
    plot class_a and class_b in histograms
    :param proj_a: the projections of class_a data onto vector w
    :param proj_b: the projections of class_b data onto vector w
    :param w: vector w
    :return: none
    """
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

    for i in range(3):
        ax[i].hist(proj_a[:, i], bins='auto', alpha=0.3, label='class a')
        ax[i].hist(proj_b[:, i], bins='auto', alpha=0.5, label='class b')
        p_title = "Projection Distribution w[" + " ".join(str(x) for x in np.around(w[:, i], 2)) + "]"
        ax[i].set_title(p_title)
        leg = ax[i].legend(loc='upper right', fancybox=True, fontsize=8)
        leg.get_frame().set_alpha(0.5)
    plt.show()


def plt_optimal_fisher_score(X_a, X_b, score_list, max_theta, max_score, w_star, Y_a, Y_b):
    """
    Plot 3 figures,
    (1) fisher score curve;
    (2) histogram of optimal classification;
    (3) optimal projection direction.
    :param X_a: Gaussian data of class a
    :param X_b: Gaussian data of class b
    :param score_list: fisher score list
    :param max_theta: theta of the optimal score
    :param max_score: optimal score
    :param w_star:  weight of the optimal score
    :param Y_a: projected class a to the weight vector of the optimal score
    :param Y_b: projected class b to the weight vector of the optimal score
    :return: none
    """
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

    # plot fisher score in terms of different projection angel
    ax[0].plot(np.linspace(-np.pi, np.pi, 360), score_list)
    ax[0].scatter(max_theta, max_score, marker='o', c='r', label='optimal')
    leg = ax[0].legend(loc='upper right', fancybox=True, fontsize=8)
    leg.get_frame().set_alpha(0.5)

    ax[0].set_ylabel("Fisher LDA Score")
    ax[0].set_xlabel("θ")
    ax[0].set_title("Fisher LDA scores by rotating angles θ")

    # plot the histogram in case of optimal classification
    ax[1].hist(Y_a, bins='auto', alpha=0.3, label='class a')
    ax[1].hist(Y_b, bins='auto', alpha=0.5, label='class b')
    p_title = "Projection Distribution w" + " ".join(str(x) for x in np.around(w_star, 2))
    ax[1].set_title(p_title)
    leg = ax[1].legend(loc='upper right', fancybox=True, fontsize=8)
    leg.get_frame().set_alpha(0.5)

    # plot the projection line
    ax[2].scatter(X_a[:, 0], X_a[:, 1], c='r', label='class a')
    ax[2].scatter(X_b[:, 0], X_b[:, 1], marker='+', c='b', label='class b')
    ax[2].set_title("Optimal projection angel for classification")
    ax[2].set_xlabel("x1")
    ax[2].set_ylabel("x2")

    w_star = np.array(w_star)
    ax[2].axis('equal')
    xielv = w_star[1]/w_star[0]

    x_point = np.linspace(-3, 3, 100)
    y_point = x_point * xielv

    ax[2].plot(x_point, y_point, label='optimal w')

    leg = plt.legend(loc='upper right', fancybox=True, fontsize=8)
    leg.get_frame().set_alpha(0.5)

    plt.show()


def plot_prob_contours(Gaus_dist_a, Gaus_dist_b, with_unbalance=False):
    """
    plot probability contours and the optimal projection line
    :param Gaus_dist_a: Gaus distribution object class a
    :param Gaus_dist_b: Gaus distribution object class b
    :param with_unbalance: if the unbalanced fisher score shall be included
    :return: none
    """

    assert(isinstance(Gaus_dist_a, model.GausDS) and isinstance(Gaus_dist_b, model.GausDS))

    X_a, X_b = Gaus_dist_a.data, Gaus_dist_b.data

    n_a = len(X_a)
    n_b = len(X_b)

    l_s_scalar_min = -9
    l_s_scalar_max = 9
    ls_x1 = np.linspace(l_s_scalar_min, l_s_scalar_max, 100)
    ls_x2 = np.linspace(l_s_scalar_min, l_s_scalar_max, 100)
    mg_x1, mg_x2 = np.meshgrid(ls_x1, ls_x2)

    pdf_a = Gaus_dist_a.Gaussian_pdf(mg_x1, mg_x2, 100)
    pdf_b = Gaus_dist_b.Gaussian_pdf(mg_x1, mg_x2, 100)

    pdf_a = pdf_a * n_a/(n_a+n_b)
    pdf_b = pdf_b * n_b/(n_a+n_b)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('equal')  # !!! axis equals so that make circle a circle

    # ax.set_title("Sa != Sb")
    ax.set_title("2 Class Classification")

    ax.scatter(X_a[:, 0], X_a[:, 1], marker='.', c='r', label='class a')
    ax.scatter(X_b[:, 0], X_b[:, 1], marker='+', c='b', label='class b')

    ax.contour(mg_x1, mg_x2, pdf_a, 10)
    ax.contour(mg_x1, mg_x2, pdf_b, 10)

    # get the decision border
    log_odds = np.log(pdf_a) - np.log(pdf_b)
    list_border = []
    for i in range(99):
        for j in range(99):
            if (log_odds[i][j]*log_odds[i][j+1] < 0) or (log_odds[i][j]*log_odds[i+1][j] < 0) \
                    or log_odds[i][j] == 0:
                list_border.append([i, j])

    bd = np.array(list_border)
    X1 = np.linspace(l_s_scalar_min, l_s_scalar_max, 100)
    Y1 = np.linspace(l_s_scalar_min, l_s_scalar_max, 100)
    ax.scatter(X1[bd[:, 0]], Y1[bd[:, 1]], marker='.', s=15, color='brown', label='decision border')

    # optimal choice of w
    init_w = np.array([1, -2]).reshape(-1, 1)

    # plot the line of w with balanced fisher score
    # equal num of points between class a and class b
    fs_clf = model.FisherScoreClassifier(X_a, X_b, init_w)
    w_star = fs_clf.classify(plot=False)
    w_star = np.array(w_star)
    xielv = w_star[1]/w_star[0]
    x_point = np.linspace(-5, 3, 100)
    y_point = x_point * xielv - 4
    plt.plot(x_point, y_point, c='g', label='optimal w')

    # plot the line of w with unbalanced fisher score
    # different num of points between class a and class b
    if with_unbalance:
        w_star = fs_clf.classify(balanced=False, plot=False)
        w_star = np.array(w_star)
        xielv = w_star[1]/w_star[0]
        x_point = np.linspace(-5, 3, 100)
        y_point = x_point * xielv - 4
        plt.plot(x_point, y_point, c='purple', label='unbalanced F(w)')

    leg = ax.legend(loc='upper right', fancybox=True, fontsize=8)
    leg.get_frame().set_alpha(0.5)

    plt.show()
