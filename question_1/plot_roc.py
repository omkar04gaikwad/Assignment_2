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
from sample_plot import Sample_validation, X_valid, labels_valid, N1_valid
from logistic_model import logistic_linear, logistic_quad

disc_erm_scores = erm_discriminant(X_valid, pdf)
roc_erm, gammas_empirical = estimate_roc(disc_erm_scores, labels_valid, N1_valid)
fig_roc = plt.figure(figsize=(10,10))
ax_roc = fig_roc.add_subplot(111)
ax_roc.plot(roc_erm['p10'], roc_erm['p11'], label="Empirical ERM Classifier ROC Curve")

ax_roc.set_xlabel(r"Probability of False Alarm $p(D=1\,|\,L=0)$")

ax_roc.set_ylabel(r"Probability of True Positive $p(D=1\,|\,L=1)$")

prob_error_empirical = np.array((roc_erm['p10'], 1 - roc_erm['p11'])).T.dot(N1_valid / Sample_validation)

min_prob_error_empirical = np.min(prob_error_empirical)
min_ind_empirical = np.argmin(prob_error_empirical)

gamma_map = pdf['prior'][0] / pdf['prior'][1]
decisions_map = disc_erm_scores >= np.log(gamma_map)

class_metrics_map = get_binary_classification_metrics(decisions_map, labels_valid, N1_valid)
min_prob_error_map = np.array((class_metrics_map['FPR'] * pdf['prior'][0] + class_metrics_map['FNR'] * pdf['prior'][1]))

ax_roc.plot(roc_erm['p10'][min_ind_empirical], roc_erm['p11'][min_ind_empirical], 'o', label='Empirical Minimum Probability: {:.3f}'.format(min_prob_error_empirical),markersize=14)
ax_roc.plot(class_metrics_map['FPR'], class_metrics_map['TPR'], '+', label='Theoretical Minimum Probability: {:.3f}'.format(min_prob_error_map), markersize=14)
ax_roc.legend()
plt.title("Empirical ERM Classifier ROC Curve")
plt.tight_layout()
plt.savefig('ERM_ROC.png')
plt.show()

print("Minimum Emperical probability: ",  "{:.4f}".format(min_prob_error_empirical))
print("Empirical Gamma: ", "{:.3f}".format(np.exp(gammas_empirical[min_ind_empirical])))

print("Minimum Theoretical probability: ",  "{:.4f}".format(min_prob_error_map))
print("Theoretical Gamma: ", "{:.3f}".format(gamma_map))


plot00 = X_valid[class_metrics_map['TN'], 0]
plot01 = X_valid[class_metrics_map['TN'], 1]
plot10 = X_valid[class_metrics_map['FP'], 0]
plot11 = X_valid[class_metrics_map['FP'], 1]
plot20 = X_valid[class_metrics_map['FN'], 0]
plot21 = X_valid[class_metrics_map['FN'], 1]
plot30 = X_valid[class_metrics_map['TP'], 0]
plot31 = X_valid[class_metrics_map['TP'], 1]

fig_disc_grid = plt.figure(figsize=(8, 8))
ax_disc = fig_disc_grid.add_subplot(111)
ax_disc.set_title(r"Validation $D^{10000}$")
ax_disc.plot(plot00, plot01, 'o', label="Correct Class 0")
ax_disc.plot(plot10, plot11, '+', label="InCorrect Class 0")
ax_disc.plot(plot20, plot21, 'x', label="InCorrect Class 1")
ax_disc.plot(plot30, plot31, 'X', label="Correct Class 1")
ax_disc.set_xlabel(r"$x_1$")
ax_disc.set_ylabel(r"$x_2$")
plot_erm(ax_disc, X_valid, pdf)
ax_disc.set_aspect('equal')
ax_disc.legend()
plt.tight_layout()
plt.savefig('Decision_Boundary_plot.png')
plt.show()

logistic_linear()
logistic_quad()