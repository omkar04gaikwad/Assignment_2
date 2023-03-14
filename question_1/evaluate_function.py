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


def erm_discriminant(X, pdf):
  class_lld_0 = (pdf['gmm_w'][0]*mvn.pdf(X,pdf['mo'][0], pdf['Co'][0]) + pdf['gmm_w'][1]*mvn.pdf(X,pdf['mo'][1], pdf['Co'][1]))
  class_lld_1 = (pdf['gmm_w'][2]*mvn.pdf(X,pdf['mo'][2], pdf['Co'][2]) + pdf['gmm_w'][3]*mvn.pdf(X,pdf['mo'][3], pdf['Co'][3]))
  erm_scores = np.log(class_lld_1) - np.log(class_lld_0)
  return erm_scores

def estimate_roc(d_score, labels, N_labels):
  sorted_score = sorted(d_score)
  gammas = ([sorted_score[0] - float_info.epsilon] + sorted_score + sorted_score[-1] + float_info.epsilon)
  decisions = [d_score >= g for g in gammas]
  ind10 = [np.argwhere((d==1) & (labels == 0)) for d in decisions]
  p10 = [len(inds) / N_labels[0] for inds in ind10]
  ind11 = [np.argwhere((d==1) & (labels == 1)) for d in decisions]
  p11 = [len(inds) / N_labels[1] for inds in ind11]
  roc = {
      'p10': np.array(p10),
      'p11': np.array(p11)
  }
  return roc, gammas

def get_binary_classification_metrics(predictions, labels, N_labels):
  class_metrics = {}
  class_metrics['TN'] = np.argwhere((predictions == 0) & (labels == 0))
  class_metrics['TNR'] = len(class_metrics['TN']) / N_labels[0]
  class_metrics['FP'] = np.argwhere((predictions == 1) & (labels == 0))
  class_metrics['FPR'] = len(class_metrics['FP']) / N_labels[0]
  class_metrics['FN'] = np.argwhere((predictions == 0) & (labels == 1))
  class_metrics['FNR'] = len(class_metrics['FN']) / N_labels[1]
  class_metrics['TP'] = np.argwhere((predictions == 1) & (labels == 1))
  class_metrics['TPR'] = len(class_metrics['TP']) / N_labels[1]

  return class_metrics

def prediction_score(X_bound, Y_bound, pdf, phi=None, num_cord=200):
  xx, yy = np.meshgrid(np.linspace(X_bound[0], X_bound[1], num_cord), np.linspace(Y_bound[0], Y_bound[1], num_cord))
  grid = np.c_[xx.ravel(), yy.ravel()]
  if phi:
    grid = phi.transform(grid)
  Z = erm_discriminant(grid, pdf).reshape(xx.shape)
  return xx, yy, Z

def plot_erm(ax, X, pdf, phi=None):
  X_bound = (floor(np.min(X[:,0])), ceil(np.max(X[:, 0])))
  Y_bound = (floor(np.min(X[:,1])), ceil(np.max(X[:, 1])))

  xx, yy, Z = prediction_score(X_bound, Y_bound, pdf, phi=None)

  equal_levels = np.array((0.3, 0.6, 0.9))
  min_Z = np.min(Z) * equal_levels[::-1]
  max_Z = np.max(Z) * equal_levels

  contour_levels = min_Z.tolist() + [0] + max_Z.tolist()
  cs = ax.contour(xx, yy, Z, contour_levels, colors='k')
  ax.clabel(cs, fontsize=14, inline=1)