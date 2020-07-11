import numpy as np
from util_funcs import *
def test_compute_cost():
    theta = np.array([[-2, -1, 1, 2]]).T

    X = np.concatenate((np.ones((5, 1)), np.reshape(np.arange(1, 16), (3, 5)).T / 10) , axis=1)
    y = np.array([1, 0, 1, 0, 1]).T

    J, grad = compute_cost(X, y, theta, lambd=3)
    
    X, y = load_txt_data('ex2data1.txt')

    one = np.ones((X.shape[0], 1))

    X = np.concatenate((one, X), axis=1)

    theta = np.array([[0, 0, 0]]).T
    J, grad = compute_cost(X, y, theta, 0)

    theta = np.array([[-24, 0.2, 0.2]]).T
    J, grad = compute_cost(X, y, theta, 0)
    print(J, grad)

if __name__ == "__main__":
    test_compute_cost()
