import numpy as np
from util_funcs import *
import matplotlib.pyplot as plt
"""Implement multi-class classification using Logistic Regression 
and One-vs-Rest method."""

# Todo:
# Read data from .mat format
# Implement LR class
# Implement one-vs-rest method.

class LogisticRegression:
    def __init__(self, X, num_classes=1):
        # self.theta = np.random.randn(X.shape[1], num_classes) * 0.1
        self.theta = np.zeros((X.shape[1], num_classes), dtype='float')

    def fit(self, X, y, num_iterations=1000, learning_rate=1):
        losses = []
        for i in range(num_iterations):
            loss, grad = compute_cost(X, y, self.theta, lambd=1)
            self.theta = update_parameters(self.theta, grad, learning_rate)
            if (i % 20 == 0):
                losses.append(loss)
        return losses

if __name__ == "__main__":
    np.random.seed(1)
    np.set_printoptions(suppress=True)

    # X, y = load_mat_data('ex3data1.mat')
    X, y = load_txt_data('ex2data2.txt')
    fig, ax = plt.subplots()
    ax = plotData(X, y, ax)
    # one = np.ones((X.shape[0], 1))
    # X = np.concatenate((one, X), axis=1)

    X = map_feature(X[:, 0, np.newaxis], X[:, 1, np.newaxis])

    lr = LogisticRegression(X)
    losses = lr.fit(X, y)

    ax = plot_decision_boundary(X, y, lr.theta, ax)
    
    ax.legend(['y = 1', 'y = 0'], loc='upper right')
    plt.title('Decision boundary')
    plt.show()