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

fig = plt.figure(figsize=(14,14))

#Sample and their labels for training for 20 samples

X_20, labels_20 = generate_data(20, pdf)
X_train.append(X_20)
labels_train.append(labels_20)
N_labels_train.append(np.array((sum(labels_20 == 0), sum(labels_20 == 1))))
ax = fig.add_subplot(221)
ax.set_title(r" Training $D^{20}$")
ax.scatter(X_20[labels_20 == 0,0], X_20[labels_20 == 0, 1],s=18, alpha=1, marker = 'o', label="Class 0")
ax.scatter(X_20[labels_20 == 1,0], X_20[labels_20 == 1, 1],s=18, alpha=1, marker = '*', label="Class 1")
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")

#Sample and their labels for training for 200 samples

X_200, labels_200 = generate_data(200, pdf)
X_train.append(X_200)
labels_train.append(labels_200)
N_labels_train.append(np.array((sum(labels_200 == 0), sum(labels_200 == 1))))
ax1 = fig.add_subplot(222)
ax1.set_title(r" Training $D^{200}$")
ax1.scatter(X_200[labels_200 == 0,0], X_200[labels_200 == 0, 1],s=18, alpha=1, marker = 'o', label="Class 0")
ax1.scatter(X_200[labels_200 == 1,0], X_200[labels_200 == 1, 1],s=18, alpha=1, marker = '*', label="Class 1")
ax1.set_xlabel(r"$x_1$")
ax1.set_ylabel(r"$x_2$")


#Sample and their labels for training for 2000 samples

X_2000, labels_2000 = generate_data(2000, pdf)
X_train.append(X_2000)
labels_train.append(labels_2000)
N_labels_train.append(np.array((sum(labels_2000 == 0), sum(labels_2000 == 1))))
ax2 = fig.add_subplot(223)
ax2.set_title(r" Training $D^{2000}$")
ax2.scatter(X_2000[labels_2000 == 0,0], X_2000[labels_2000 == 0, 1],s=18, alpha=1, marker = 'o', label="Class 0")
ax2.scatter(X_2000[labels_2000 == 1,0], X_2000[labels_2000 == 1, 1],s=18, alpha=1, marker = '*', label="Class 1")
ax2.set_xlabel(r"$x_1$")
ax2.set_ylabel(r"$x_2$")
ax.legend()
ax1.legend()
ax2.legend()

#Sample and their labels for Validation for 10000 samples

Sample_validation = 10000
X_valid, labels_valid = generate_data(Sample_validation, pdf)
N1_valid = np.array((sum(labels_valid == 0), sum(labels_valid == 1)))
ax3 = fig.add_subplot(224)
ax3.set_title(r"Validation $D^{10000}$")
ax3.scatter(X_valid[labels_valid==0, 0], X_valid[labels_valid==0, 1],s=18, alpha=1, marker = 'o', label="Class 0")
ax3.scatter(X_valid[labels_valid==1, 0], X_valid[labels_valid==1, 1],s=18, alpha=1, marker = '*', label="Class 1")
ax3.set_xlabel(r"$x_1$")
ax3.set_ylabel(r"$x_2$")
ax3.legend()

x1_valid_lim = (floor(np.min(X_valid[:,0])), ceil(np.max(X_valid[:,0])))
x2_valid_lim = (floor(np.min(X_valid[:,1])), ceil(np.max(X_valid[:,1])))

plt.setp(ax, xlim=x1_valid_lim, ylim=x2_valid_lim)
plt.tight_layout()
plt.savefig('Sample_question_1.png')
plt.show()
