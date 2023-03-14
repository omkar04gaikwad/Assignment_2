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
fig = plt.figure(figsize=(12,12))



def generateDataFromGMM(N,gmm_pdf):
    u = np.random.random(N)
    thresholds = np.cumsum(gmm_pdf['priors'])
    thresholds = np.insert(thresholds, 0, 0)  # For intervals of classes

    n = gmm_pdf['meanVectors'].shape[0]  # Data dimensionality

    X = np.zeros((N, n))
    C = len(gmm_pdf['priors'])  # Number of components
    for i in range(C + 1):
        # Get randomly sampled indices for this Gaussian, checking between thresholds based on class priors
        indices = np.argwhere((thresholds[i - 1] <= u) & (u <= thresholds[i]))[:, 0]
        # No. of samples in this Gaussian
        X[indices, :] = mvn.rvs(gmm_pdf['meanVectors'][i - 1], gmm_pdf['covMatrices'][i - 1], len(indices))

    return X[:, 0:2], X[:, 2]


def generate_data(N):
    pdf = {
        'priors' : [.3,.4,.3],
        'meanVectors' : np.array([[-10, 0, 10], [0, 0, 0], [10, 0, -10]]),
        'covMatrices' : np.array([[[1, 0, -3], [0, 1, 0], [-3, 0, 15]], [[8, 0, 0], [0, .5, 0], [0, 0, .5]], [[1, 0, -3], [0, 1, 0], [-3, 0, 15]]])
    }
    x,labels = generateDataFromGMM(N,pdf)
    return x, labels

def plot21(a,b,c,mark="o",col="b"):
  ax = fig.add_subplot(121, projection='3d')
  ax.scatter(a, b, c,marker=mark,color=col)
  ax.set_xlabel("x1")
  ax.set_ylabel("x2")
  ax.set_zlabel("y")
  ax.set_title('Training Dataset for 100 samples')

def plot22(a,b,c,mark="+",col="r"):
  ax = fig.add_subplot(122, projection='3d')
  ax.scatter(a, b, c,marker=mark,color=col)
  ax.set_xlabel("x1")
  ax.set_ylabel("x2")
  ax.set_zlabel("y")
  ax.set_title('Validate Dataset for 1000 samples')

def hw2q2():
    Ntrain = 100
    xTrain,yTrain = generate_data(Ntrain)
    plot21(xTrain[:,0],xTrain[:,1],yTrain)
    Ntrain = 1000
    xValidate,yValidate = generate_data(Ntrain)
    plot22(xValidate[:,0],xValidate[:,1],yValidate)
    plt.savefig("Scatterplot_dataset.png")
    plt.legend()
    plt.show()
    
    
    return xTrain,yTrain,xValidate,yValidate

def prediction_score(X_bound, Y_bound, pdf, prediction_function, phi=None, num_cord=100):
  xx, yy = np.meshgrid(np.linspace(X_bound[0], X_bound[1], num_cord), np.linspace(Y_bound[0], Y_bound[1], num_cord))
  grid = np.c_[xx.ravel(), yy.ravel()]
  if phi:
    grid = phi.transform(grid)
  Z = prediction_function(grid, pdf).reshape(xx.shape)
  return xx, yy, Z