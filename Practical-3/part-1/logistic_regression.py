import numpy as np

class LogisticRegression(object):
    
    def __init__(self, epsilon=0.0001, l=1, step_size=0.01, max_steps=1000, initial_beta=None):
        self.epsilon = epsilon
        self.l = l
        self.step_size = step_size
        self.max_steps = max_steps
        self.initial_beta = initial_beta

    def fit(self, X, Y):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array
        """
        self.beta = stochastic_gradient_descent(X = X, 
                                                Y = Y, 
                                                epsilon=self.epsilon, 
                                                l=self.l, 
                                                step_size=self.step_size, 
                                                max_steps=self.max_steps,
                                               initial_beta=self.initial_beta)

    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: Y - 1 dimension python list with labels
        """
        Y = []
        for row in X:
            if np.dot(row, self.beta) >= 0:
                Y.append(1)
            else:
                Y.append(0)
        return Y

def sigmoid(s):
    return 1/(1+np.exp(-s))

def cost_function(X, Y, beta):
    h = sigmoid(np.dot(X, beta))
    return (-np.dot(Y, np.log(h))+ np.dot((1+Y),1-h))/X.shape[0]

def stochastic_gradient_descent(X, Y, epsilon=0.0001, l=1, step_size=0.01, max_steps=1000, initial_beta=None):
    """
    Implement gradient descent using stochastic approximation of the gradient.
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param l: regularization parameter lambda
    :param epsilon: approximation strength
    :param max_steps: maximum number of iterations before algorithm will
        terminate.
    :return: value of beta (1 dimensional np.array)
    """
    N, D = X.shape[0], X.shape[1]
    X, l_vector, var_cols, std_cols, mean_cols = parameters_for_scaling(X, l)
    l_vector[0] = 0
    if initial_beta == None:
        beta = np.zeros(D)
    else:
        beta = initial_beta
    for s in range(max_steps):
        if s % N == 0:
            X, Y = shuffle_data(X, Y)
        next_beta = beta - step_size*normalized_gradient(X[s%N], Y[s%N], beta, l_vector)
        if s % 1000 == 0:
            print('iteration= {}, and cost_value= {}'.format(s, cost_function(X, Y, beta)))
        if np.linalg.norm(next_beta - beta)/np.linalg.norm(next_beta) < epsilon:
            print('the gradient descent algorithm is finished.')
            return get_real_beta(next_beta, std_cols, mean_cols)
        beta = next_beta
    print('the gradient descent algorithm is finished.')
    return get_real_beta(beta, std_cols, mean_cols)



def normalized_gradient(X, Y, beta, l):
    """
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param beta: value of beta (1 dimensional np.array)
    :param l: regularization parameter lambda
    :return: normalized gradient, i.e. gradient normalized according to data
    """
    return (np.dot(X.T, sigmoid(np.dot(X,beta))-Y) + l*beta/2)/X.shape[0]

def get_real_beta(beta, std_cols, mean_cols):
    beta = np.copy(beta)
    beta[1:] = beta[1:]/std_cols[1:]
    beta[0] = beta[0] - np.dot(mean_cols[1:], beta[1:])
    return beta

def parameters_for_scaling(X, l):
    X = np.copy(X)
    featchures_mat = X[:, 1:]
    var_cols = np.var(X, axis=0)
    std_cols = np.std(X, axis=0)
    mean_cols = np.mean(X, axis=0)
    var_cols[0] = 1
    featchures_mat = X[:, 1:]
    X[:, 1:] = div0(featchures_mat - np.mean(featchures_mat, axis=0),np.std(featchures_mat, axis=0))
    return X, div0(l,var_cols), var_cols, std_cols, mean_cols

def shuffle_data(X, Y):
    data = np.hstack((X, np.array(Y).reshape(len(Y), 1)))
    np.random.shuffle(data) 
    return data[:,:X.shape[1]], data[:,X.shape[1]:].reshape(len(Y))

def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c