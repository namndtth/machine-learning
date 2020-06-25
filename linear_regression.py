# Implement linear regression from scratch by using numpy.
# Reuse data from machine learning course of Andrew Ng

import numpy as np
import matplotlib.pyplot as plt

def load_data(fpath):
    """
    Params:
        fpath: string that is path to data file.
    Returns:
        (X, Y): tuple of numpy arrays.
                X is input and Y is label, both shape (num_examples, 1).
    """

    with open(fpath, "r") as f:
        data = f.readlines()

    # Number of training examples
    m = len(data)

    X = np.zeros((m, 1))
    Y = np.zeros((m, 1))

    for idx in range(0, m):
        x, y = data[idx].split(",")
        X[idx], Y[idx] = float(x), float(y)
    
    # Add columns of one to X
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    return (X, Y)

def visualize(X, Y):
    """Visualize X and Y.
    Params: 
        X (numpy array): input.
        Y (numpy array): label.
    """
    plt.plot(X, Y, "bx")
    plt.show()

def compute_cost(X, Y, theta):
    """Compute cost function for over m training examples.
    Params:
        X (numpy array): input.
        Y (numpy array): label.
        theta (numpy array): learnable parameters.
    Returns:
        J: cost value.
    """
    m = len(X)
    y_hat = np.zeros((m, 1))
    
    # Using loops for calcuating y_hat
    # for i in range(0, m):
    #     y_hat[i] = np.dot(theta.T, X[i])

    # Avoid loops
    y_hat = np.sum(theta.T * X, axis=1, keepdims=True)

    J = np.sum(np.square(y_hat - Y)) / (2 * m)

    return J

def fprime(X, Y, theta):
    # The derivative of h_theta(x)
    m = X.shape[0]
    y_hat = np.sum(theta.T * X, axis=1, keepdims=True)
    grad = np.sum(X * (y_hat - Y), axis=0, keepdims=True) / m
    return grad.T

def fprime_approx(X, Y, theta, eps=1e-4):
    # Numerical gradient for checking the derivative
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        w_p = theta.copy()
        w_n = theta.copy()
        w_p[i] += eps
        w_n[i] -= eps
        grad[i] = (compute_cost(X, Y, w_p) - compute_cost(X, Y, w_n)) / (2 * eps)
    return grad

def gradient_descent(X, Y, theta, learning_rate, num_iterations):
    # Using gradient descent algorithms and update parameters
    theta_history = []
    J_history = []
    diff = []
    for _ in range(num_iterations):

        grad = fprime(X, Y, theta)
  
        grad_approx = fprime_approx(X, Y, theta)

        enum = np.linalg.norm(grad - grad_approx)
        denom = np.linalg.norm(grad) + np.linalg.norm(grad_approx)

        diff.append(enum / denom)
        theta = theta - learning_rate * grad

        J = compute_cost(X, Y, theta)
        J_history.append(J)
        theta_history.append(theta)

    return J_history, theta_history

def 
if __name__ == "__main__":

    # Our hypothesis is h_theta(x) = w0 * x0 + w1 * x1
    # Cost function L(w0, w1) = 1 / (2 * m) * (y_hat - y) ^ 2 = 1 / (2 * m) * (w0 * x0 + w1 * x1 - y) ^ 2
    # Derivative of loss function w.r.t parameters
    # dL/dw0 = 1 / m  * x0 * (w0 * x0 + w1 * x1 - y)
    # dL/dw1 = 1 / m  * x1 * (w0 * x0 + w1 * x1 - y)
    X, Y = load_data("ex1data1.txt")
    np.random.seed(0)
    
    X = np.random.rand(1000, 1)
    Y = 4 + 3 * X + .2 * np.random.randn(1000, 1) # noise added
    # visualize(X, Y)
    
    one = np.ones((X.shape[0], 1))
    X = np.concatenate((one, X), axis = 1)
    
    theta = np.array([[-1.], [2.]], dtype=np.float32)
    J_history, theta_history = gradient_descent(X, Y, theta, 0.1, 1000)

    theta_final = theta_history[-1]
    x0 = np.linspace(0, 1, 2, endpoint=True)

    y0 = theta[0] + theta[1] * x0
    
    print(theta)
    
    plt.figure()
    plt.subplot(211)
    plt.plot(X[:, 1], Y, 'bo')
    plt.plot(x0, y0, 'y', linewidth=2)

    plt.subplot(212)
    plt.plot(J_history)
    plt.xlabel("Num iterations")
    plt.ylabel("Loss")
    plt.show()

