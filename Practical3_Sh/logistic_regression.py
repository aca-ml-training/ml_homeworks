import numpy as np


def normalize(X):

    mean = np.zeros(X.shape[1])
    sd = np.zeros(X.shape[1])

    for i in range(1, X.shape[1]):
        m = np.mean(X[:,i])
        s = np.std(X[:,i])
        X[:,i] = (X[:,i] - m)/s
        mean[i] = m
        sd[i] = s
        
    return mean, sd, X

def denormalize(beta, mean, sd):
    s1 = mean[1] * beta[1] / sd[1]
    s2 = mean[2] * beta[2] / sd[2]

    beta_new = np.empty(shape=beta.shape)
    beta_new[0] = beta[0] - s1 - s2
    beta_new[1] = beta[1]/s1
    beta_new[2] = beta[2]/s2
    return beta_new


def sigmoid(X):
    return 1 / (1 + np.math.exp(-X))

def normalized_gradient(X, Y, beta, l):
    
    """
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param beta: value of beta (1 dimensional np.array)
    :param l: regularization parameter lambda
    :return: normalized gradient, i.e. gradient normalized according to data
    """
    
    gr = logistic_func(X, beta) - np.squeeze(Y)
    gr = gr.T.dot(x)/Y.size - beta.dot(l)/Y.size
    
    return gr

def logistic_func(beta, x):
    return float(1) / (1 + np.math.e**(-x.dot(beta)))

def log_gradient(beta, x, y):
    gr = (logistic_func(beta, x) - np.squeeze(y)).T.dot(x)
    return gr


def cost_func(beta, x, y):
    
    y = np.squeeze(y)
    final = (-(y * np.log(logistic_func(beta,x))) - (1-y) * np.log(1 - logistic_func(beta,x)))
    return np.mean(final)



def gradient_descent(X, Y, epsilon=1e-6, l=1, step_size=1e-4, max_steps=1000):
    
    """
    Implement gradient descent using full value of the gradient.
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param l: regularization parameter lambda
    :param epsilon: approximation strength
    :param max_steps: maximum number of iterations before algorithm will
        terminate.
    :return: value of beta (1 dimensional np.array)
    """
    
    mean, sd, X = normalize(X)
    
    beta = np.zeros(X.shape[1])
   
    
    
    lsc = np.zeros(len(beta))
    for i in range(len(beta)):
        lsc[i] = beta[i]*l / (sd[i]**2)
    lsc[0] = 0
    
 
    cost = cost_func(beta, X, Y)
    change_cost = 1

    while(change_cost > epsilon):
        old_cost = cost
        beta = beta - (step_size * log_gradient(beta, X, Y)) + step_size*lsc
        cost = cost_func(beta, X, Y)
        change_cost = old_cost - cost
       
    

    beta = denormalize(beta, mean, sd)
 
    return beta
