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
from estimator import map_estimator, mle_estimator, ase
from sample import hw2q2, prediction_score

xTrain,yTrain,xValidate,yValidate = hw2q2()

phi = PolynomialFeatures(degree=3)
X_train_cubic = phi.fit_transform(xTrain)
theta_mle = mle_estimator(X_train_cubic, yTrain)

X_valid_cubic = phi.transform(xValidate)
y_pred_mle = X_valid_cubic.dot(theta_mle)

ase_mle = ase(y_pred_mle, yValidate)
print("Average Squared-Error on Validation set for ML estimator = ", "{:.4f}".format(ase_mle))
x1_valid_lim = (floor(np.min(xValidate[:,0])), ceil(np.max(xValidate[:,0])))
x2_valid_lim = (floor(np.min(xValidate[:,1])), ceil(np.max(xValidate[:,1])))

reg_fun = lambda X, th: X.dot(th)
xx, yy, Z = prediction_score(x1_valid_lim, x2_valid_lim, theta_mle, reg_fun, phi, num_cord=100)


fig_mle = plt.figure(figsize=(10, 10))
ax_mle = fig_mle.add_subplot(111, projection ='3d')

# Plot the best fit plane on the 2D real vector samples
ax_mle.scatter(xValidate[:,0], xValidate[:,1], yValidate, marker='+', color='r')
ax_mle.plot_surface(xx, yy, Z, color='blue', alpha=0.3)
ax_mle.set_xlabel(r"$x_1$")
ax_mle.set_ylabel(r"$x_2$")
ax_mle.set_zlabel(r"$y$")
ax_mle.text2D(0.05, 0.95, "Average Squared-Error on Validation set: %.3f" % ase_mle, transform=ax_mle.transAxes)
plt.title("Maximum Likelihood Estimator on Validation Set of 1000 samples")
plt.tight_layout()
plt.savefig("Scatterplot_mlestimator.png")
plt.show()



gamma_sample = 1000
gammas = np.geomspace(10**-7, 10**7, num=gamma_sample)
average_se_map = np.empty(gamma_sample) 
for i, gam in enumerate(gammas):
    Map_theta = map_estimator(X_train_cubic, yTrain, gam)
    y_pred_map = X_valid_cubic.dot(Map_theta)
    average_se_map[i] = ase(y_pred_map, yValidate)

out_string = "Best Average Squared-Error for MAP estimator for gamma = " + str(gammas[np.argmin(average_se_map)]) + " is : " + str("{:.3f}".format(np.min(average_se_map)))
print(out_string)

fig_map = plt.figure(figsize=(10,10))
ax_map = fig_map.add_subplot(111)
ax_map.plot(gammas,average_se_map, color='b', label=r"$\gamma_{MAP}$")
plt.axhline(y=ase_mle, xmin=10**-7, xmax=10**7, color='red',label=r"$\gamma_{MLE}$")

ax_map.set_xscale('log')
ax_map.set_xticks(np.geomspace(10**-7, 10**7, num=15))

ax_map.set_xlabel(r"$\gamma$")
ax_map.set_ylabel(r"$ASE_{valid}$")
ax_map.set_title("Maximum-a-Posteriori on Validation Set of 1000 samples")
plt.savefig("Scatterplot_mapstimator.png")
plt.legend()
plt.show() 