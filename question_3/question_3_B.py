import numpy as np
import matplotlib.pyplot as plt

# True position of Vehicle
true_vehicle_position = np.array([0.4, 0.8])

# Noise measurement range
noiserange = 0.3

# range measurement model
def range_model(point):
    value_1 = np.sqrt(np.sum((point - true_vehicle_position)**2)) + np.random.normal(0, noiserange)
    return value_1

# objective function for MAP estimator
def objective_function(point, landmark, ranges):
    prior = np.sum((point - np.array([0, 0]))**2) / 2
    error = np.sum((ranges - np.sqrt(np.sum((landmark - point)**2, axis=1)))**2)
    val = prior + error
    return val

# sample rejection measurement range
def sample_range(landmark):
    while True:
        sample = range_model(landmark)
        if sample >= 0:
            return sample

def plot_contour(K):
    # place landmarks on circle
    theta = np.linspace(0, 2*np.pi, K+1)[:-1]
    landmarks = np.array([np.cos(theta), np.sin(theta)]).T
    # Generate range
    ranges = np.array([sample_range(landmark) for landmark in landmarks])
    # Define range for coordinates
    x_range = y_range = np.linspace(-2, 2, 101)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)

    #Compute objective function
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            point = np.array([x_range[i], y_range[j]])
            Z[j, i] = objective_function(point, landmarks, ranges)
    
    # Plot contours of Objective Function
    plt.figure(figsize=(10, 10))
    levels = np.logspace(np.log10(3/2), np.log10(5/2), 5)
    plt.contour(X, Y, Z, levels=levels)
    plt.plot(true_vehicle_position[0], true_vehicle_position[1], 'bx', markersize=12, label=f'True Vehicle position: {true_vehicle_position}')
    plt.plot(landmarks[:, 0], landmarks[:, 1], 'ro', markersize=8, label=f'K value position: {landmarks}')
    plt.axis('equal')
    plt.legend()
    plt.title(f'K = {K}')
    plt.savefig(f"K_{K}.png")
    plt.show()

for K in range(1,5):
    plot_contour(K)

