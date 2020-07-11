import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def compute_cost(X, y, theta, lambd):
    """Compute cost and gradient for logistic regression with regularization over m training examples"""
    assert(theta.shape[0] == X.shape[1])
    
    m = X.shape[0]
    grad = np.zeros(y.shape)
    J = 0
    
    output = sigmoid(np.dot(X, theta))

    J = np.sum(- y * np.log(output) - (1 - y) * np.log(1 - output)) / m + lambd / (2 * m) * np.sum(np.square(theta[1:]))

    grad = np.dot(X.T, (output - y)) / m
    
    grad[1:] = grad[1:] + lambd / m * theta[1:]

    return J, grad

def update_parameters(theta, grad, learning_rate):
    theta = theta - learning_rate * grad
    return theta

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def load_mat_data(file_name):
    '''Load data from MATLAB file
    Params:
        file_name: str
            Name of mat file
    Returns:
        X: numpy array shape [m, n]. m: number of examples. n: number of features.
            Input data.
        y: numpy array shape [m, 1].
            Ground truth.

    '''
    data = sio.loadmat(file_name)
    X, y = data['X'], data['y']
    return X, y

def load_txt_data(file_name):
    data = np.loadtxt(file_name, delimiter=',')

    X, y = data[:,:-1], data[:, -1:]

    return X, y    

def visualize_handwritten_digit(X):
    """Visualize MNIST dataset."""
    image = X[0].reshape((20, 20))

    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.show()

def plotData(X, y, ax):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    
    ax.plot(X[pos, 0], X[pos, 1], "k+", linewidth=2, markersize=7)
    ax.plot(X[neg, 0], X[neg, 1], "ko", mfc='y', markersize=7)
    ax.set_xlabel('Microchip Test 1')
    ax.set_ylabel('Microchip Test 2')

    return ax
def map_feature(X1, X2):
    """Combine feature to create polynomio to power of 6."""
    degree = 6
    out = np.ones(X1.shape)
    
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            out = np.concatenate((out, np.power(X1, (i - j)) * np.power(X2, j)), axis=-1)

    return out

def plot_decision_boundary(X, y, theta, ax):
    
    u = np.linspace(-1, 1.5, 50)
    
    v = np.linspace(-1, 1.5, 50)

    z = np.zeros((len(u), len(v)))

    for i in range(0, len(u)):
        for j in range(0, len(v)):
            z[i, j] = np.dot(map_feature(u[i, np.newaxis], v[j, np.newaxis]), theta)

    z = z.T
    
    ax.contour(u, v, z, [0, 1e-3], linewidths=2)
    return ax
    