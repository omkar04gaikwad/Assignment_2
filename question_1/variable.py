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

pdf = {
    'prior': np.array([0.6, 0.4]),
    'gmm_w': np.array([0.5, 0.5, 0.5, 0.5]),
    'mo': np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]]),
    'Co': np.array([[[1,0], [0, 1]],
                    [[1,0], [0, 1]],
                    [[1,0], [0, 1]],
                    [[1,0], [0, 1]]])
}

X_train = []
labels_train = []
N_labels_train = []

def generate_data(N, pdf):
  n = pdf['mo'].shape[1]
  u = np.random.rand(N)
  thresholds = np.cumsum(np.append(pdf['gmm_w'].dot(pdf['prior'][0]), pdf['prior'][1]))
  thresholds = np.insert(thresholds, 0, 0)
  labels = u >= pdf['prior'][0]
  X = np.zeros((N, n))
  guass = len(pdf['mo'])
  for i in range(0, guass):
    indice = np.argwhere((thresholds[i-1] <= u) & (u <= thresholds[i]))[:, 0]
    X[indice, :] = mvn.rvs(pdf['mo'][i-1], pdf['Co'][i-1], len(indice))
  return X, labels