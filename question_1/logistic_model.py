from sys import float_info
from math import ceil, floor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import multivariate_normal as mvn
from sklearn.preprocessing import PolynomialFeatures
np.set_printoptions(suppress=True)
np.random.seed(7)
from variable import pdf, generate_data, X_train, labels_train, N_labels_train
from evaluate_function import erm_discriminant, estimate_roc, get_binary_classification_metrics, plot_erm
from sample_plot import Sample_validation, X_valid, labels_valid, N1_valid, x1_valid_lim, x2_valid_lim

Epi = 1e-7

def logistic_prediction(X, w):
  logits = X.dot(w)
  y = 1 + np.exp(-logits)
  value = 1.0/y
  return value

def negative_log_likelihood(labels, predictions):
  predict1 = np.clip(predictions, Epi, 1-Epi)
  log_p0 = (1-labels)*np.log(1 - predict1 + Epi)
  log_p1 = labels * np.log(predict1 + Epi)
  return -np.mean(log_p0 + log_p1, axis=0)

def Compute_param_logistic(X, labels):
  theta0 = np.random.randn(X.shape[1])
  cost_fun = lambda w: negative_log_likelihood(labels, logistic_prediction(X, w))
  res = minimize(cost_fun, theta0, tol=1e-6)
  return res.x

def prediction_score_grid(X_bound, Y_bound, pdf, phi=None, num_cord=200):
  xx, yy = np.meshgrid(np.linspace(X_bound[0], X_bound[1], num_cord), np.linspace(Y_bound[0], Y_bound[1], num_cord))
  grid = np.c_[xx.ravel(), yy.ravel()]
  if phi:
    grid = phi.transform(grid)
  Z = logistic_prediction(grid, pdf).reshape(xx.shape)
  return xx, yy, Z

def logistic_classifier(ax, X, w, labels, N_labels, phi=None):
  predictions = logistic_prediction(phi.fit_transform(X), w)
  decisions = np.array(predictions >= 0.5)
  logistic_metrics = get_binary_classification_metrics(decisions, labels, N_labels)
  probability_error = np.array((logistic_metrics['FPR'], logistic_metrics['FNR'])).T.dot(N_labels / labels.shape[0])
  ax.plot(X[logistic_metrics['TN'], 0], X[logistic_metrics['TN'], 1], 'og', label="Correct Class 0");
  ax.plot(X[logistic_metrics['FP'], 0], X[logistic_metrics['FP'], 1], 'or', label="Incorrect Class 0");
  ax.plot(X[logistic_metrics['FN'], 0], X[logistic_metrics['FN'], 1], '+r', label="Incorrect Class 1");
  ax.plot(X[logistic_metrics['TP'], 0], X[logistic_metrics['TP'], 1], '+g', label="Correct Class 1");
  xx, yy, Z = prediction_score_grid(x1_valid_lim, x2_valid_lim, w, phi)
  cs = ax.contour(xx, yy, Z, levels=1, colors='k')
  ax.set_xlabel(r"$x_1$")
  ax.set_ylabel(r"$x_2$")
  return probability_error

def logistic_linear():
    print("Linear Logistic Model")
    fig_linear = plt.figure(figsize=(15, 15))
    ax_linear_20 = fig_linear.add_subplot(321)
    ax_linear_200 = fig_linear.add_subplot(323)
    ax_linear_2000 = fig_linear.add_subplot(325)
    ax_linear_422 = fig_linear.add_subplot(322)
    ax_linear_424 = fig_linear.add_subplot(324)
    ax_linear_426 = fig_linear.add_subplot(326)
    phi = PolynomialFeatures(degree=1)

    # 20 samples
    w_mle_20 = Compute_param_logistic(phi.fit_transform(X_train[0]), labels_train[0])
    probability_error_20 = logistic_classifier(ax_linear_20, X_train[0], w_mle_20, labels_train[0], N_labels_train[0], phi)
    probability_error_valid_20 = logistic_classifier(ax_linear_422, X_valid, w_mle_20, labels_valid, N1_valid, phi)
    print("Linear Logistic Model for Train Sample = 20 MLE for w: ", w_mle_20)
    print("Training set error for Sample = 20 classifier error = ","{:.4f}".format(probability_error_20))
    print("Validation set error for Sample = 20 classifier error = ", "{:.4f}".format(probability_error_valid_20))
    ax_linear_20.set_title("Decision Boundary for Linear Logisitc Model training Sample = 20")
    ax_linear_20.set_xticks([])
    ax_linear_422.set_title("Decision Boundary for Linear Logisitc Model Validation Sample = 10000")
    ax_linear_422.set_xticks([])

    # 200 samples
    w_mle_200 = Compute_param_logistic(phi.fit_transform(X_train[1]), labels_train[1])
    probability_error_200 = logistic_classifier(ax_linear_200, X_train[1], w_mle_200, labels_train[1], N_labels_train[1], phi)
    probability_error_valid_200 = logistic_classifier(ax_linear_424, X_valid, w_mle_200, labels_valid, N1_valid, phi)
    print("Linear Logistic Model for Train Sample = 200 MLE for w: ", w_mle_200)
    print("Training set error for Sample = 200 classifier error = ","{:.4f}".format(probability_error_200))
    print("Validation set error for Sample = 200 classifier error = ", "{:.4f}".format(probability_error_valid_200))
    ax_linear_200.set_title("Decision Boundary for Linear Logisitc Model training Sample = 200")
    ax_linear_200.set_xticks([])
    ax_linear_424.set_title("Decision Boundary for Linear Logisitc Model Validation Sample = 10000")
    ax_linear_424.set_xticks([])

    # 2000 samples
    w_mle_2000 = Compute_param_logistic(phi.fit_transform(X_train[2]), labels_train[2])
    probability_error_2000 = logistic_classifier(ax_linear_2000, X_train[2], w_mle_2000, labels_train[2], N_labels_train[2], phi)
    probability_error_valid_2000 = logistic_classifier(ax_linear_426, X_valid, w_mle_2000, labels_valid, N1_valid, phi)
    print("Linear Logistic Model for Train Sample = 2000 MLE for w: ", w_mle_2000)
    print("Training set error for Sample = 20 classifier error = ","{:.4f}".format(probability_error_2000))
    print("Validation set error for Sample = 20 classifier error = ", "{:.4f}".format(probability_error_valid_2000))
    ax_linear_2000.set_title("Decision Boundary for Linear Logisitc Model training Sample = 2000")
    ax_linear_2000.set_xticks([])
    ax_linear_426.set_title("Decision Boundary for Linear Logisitc Model Validation Sample = 10000")
    ax_linear_426.set_xticks([])

    handles, labels = ax_linear_20.get_legend_handles_labels()
    fig_linear.legend(handles, labels, loc='lower center')
    plt.setp(ax_linear_20, xlim=x1_valid_lim, ylim=x2_valid_lim)
    plt.savefig('linear_logistic.png')
    plt.show()


def logistic_quad():
    print("Quadratic Logistic Model")
    fig_linear = plt.figure(figsize=(15, 15))
    ax_linear_20 = fig_linear.add_subplot(321)
    ax_linear_200 = fig_linear.add_subplot(323)
    ax_linear_2000 = fig_linear.add_subplot(325)
    ax_linear_422 = fig_linear.add_subplot(322)
    ax_linear_424 = fig_linear.add_subplot(324)
    ax_linear_426 = fig_linear.add_subplot(326)
    phi = PolynomialFeatures(degree=2)

    # 20 samples
    w_mle_20 = Compute_param_logistic(phi.fit_transform(X_train[0]), labels_train[0])
    probability_error_20 = logistic_classifier(ax_linear_20, X_train[0], w_mle_20, labels_train[0], N_labels_train[0], phi)
    probability_error_valid_20 = logistic_classifier(ax_linear_422, X_valid, w_mle_20, labels_valid, N1_valid, phi)
    print("Linear Logistic Model for Train Sample = 20 MLE for w: ", w_mle_20)
    print("Training set error for Sample = 20 classifier error = ","{:.4f}".format(probability_error_20))
    print("Validation set error for Sample = 20 classifier error = ", "{:.4f}".format(probability_error_valid_20))
    ax_linear_20.set_title("Decision Boundary for Linear Logisitc Model training Sample = 20")
    ax_linear_20.set_xticks([])
    ax_linear_422.set_title("Decision Boundary for Linear Logisitc Model Validation Sample = 10000")
    ax_linear_422.set_xticks([])

    # 200 samples
    w_mle_200 = Compute_param_logistic(phi.fit_transform(X_train[1]), labels_train[1])
    probability_error_200 = logistic_classifier(ax_linear_200, X_train[1], w_mle_200, labels_train[1], N_labels_train[1], phi)
    probability_error_valid_200 = logistic_classifier(ax_linear_424, X_valid, w_mle_200, labels_valid, N1_valid, phi)
    print("Linear Logistic Model for Train Sample = 200 MLE for w: ", w_mle_200)
    print("Training set error for Sample = 200 classifier error = ","{:.4f}".format(probability_error_200))
    print("Validation set error for Sample = 200 classifier error = ", "{:.4f}".format(probability_error_valid_200))
    ax_linear_200.set_title("Decision Boundary for Linear Logisitc Model training Sample = 200")
    ax_linear_200.set_xticks([])
    ax_linear_424.set_title("Decision Boundary for Linear Logisitc Model Validation Sample = 10000")
    ax_linear_424.set_xticks([])

    # 2000 samples
    w_mle_2000 = Compute_param_logistic(phi.fit_transform(X_train[2]), labels_train[2])
    probability_error_2000 = logistic_classifier(ax_linear_2000, X_train[2], w_mle_2000, labels_train[2], N_labels_train[2], phi)
    probability_error_valid_2000 = logistic_classifier(ax_linear_426, X_valid, w_mle_2000, labels_valid, N1_valid, phi)
    print("Linear Logistic Model for Train Sample = 2000 MLE for w: ", w_mle_2000)
    print("Training set error for Sample = 20 classifier error = ","{:.4f}".format(probability_error_2000))
    print("Validation set error for Sample = 20 classifier error = ", "{:.4f}".format(probability_error_valid_2000))
    ax_linear_2000.set_title("Decision Boundary for Linear Logisitc Model training Sample = 2000")
    ax_linear_2000.set_xticks([])
    ax_linear_426.set_title("Decision Boundary for Linear Logisitc Model Validation Sample = 10000")
    ax_linear_426.set_xticks([])

    handles, labels = ax_linear_20.get_legend_handles_labels()
    fig_linear.legend(handles, labels, loc='lower center')
    plt.setp(ax_linear_20, xlim=x1_valid_lim, ylim=x2_valid_lim)
    plt.savefig('Quadratic_logistic.png')
    plt.show()