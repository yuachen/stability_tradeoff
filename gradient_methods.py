import numpy as np


def nesterov(x0, grad, step_size, T=1000):
    lambdas = np.zeros(T)
    rs = np.zeros(T)
    for s in range(1, T):
        lambdas[s] = (1.0+np.sqrt(1+4*lambdas[s-1]*lambdas[s-1]))/2.0
        rs[s-1] = (1.0-lambdas[s-1])/lambdas[s]

    m = np.shape(x0)[0]
    xs = np.zeros((T, m))
    xs[0] = x0
    ys = np.zeros((T, m))
    for i in range(1, T):
        ys[i] = xs[i-1] - step_size * grad(xs[i-1])
        xs[i] = (1-rs[i])*ys[i] + rs[i]*ys[i-1]

    return (xs, ys)

def heavy_ball(x0, grad, step_size, gamma, T=1000):
    m = np.shape(x0)[0]
    xs = np.zeros((T, m))
    xs[0] = x0
    xs[1] = x0
    for i in range(2, T):
        xs[i] = xs[i-1] - step_size * grad(xs[i-1]) + gamma * (xs[i-1] - xs[i-2])
        
    return xs
    

def fg(x0, grad, step_size, T=1000):
    m = np.shape(x0)[0]
    xs = np.zeros((T, m))
    xs[0] = x0
    for i in range(1, T):
        xs[i] = xs[i-1] - step_size * grad(xs[i-1])

    return xs

def fg_decrease(x0, grad, step_size, alpha, T=1000):
    m = np.shape(x0)[0]
    xs = np.zeros((T, m))
    xs[0] = x0
    for i in range(1, T):
        xs[i] = xs[i-1] - step_size*i**(-alpha) * grad(xs[i-1])

    return xs


def sgd(x0, grad, n, step_size, T=1000, seed = 123456):
    np.random.seed(seed)
    m = np.shape(x0)[0]
    xs = np.zeros((T, m))
    xs[0] = x0
    for i in range(1, T):
        index = np.random.choice(n)
        xs[i] = xs[i-1] - step_size * grad(xs[i-1], index)

    return xs

def sgd_decrease(x0, grad, n, step_size, alpha, T=1000, seed = 123456):
    np.random.seed(seed)
    m = np.shape(x0)[0]
    xs = np.zeros((T, m))
    xs[0] = x0
    for i in range(1, T):
        index = np.random.choice(n)
        xs[i] = xs[i-1] - step_size * i**(-alpha) * grad(xs[i-1], index)

    return xs