from numpy import asfortranarray, squeeze, asarray, copy

from gradientFunction import gradientFunction


def gradientFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    m = len(y)   # number of training examples

    theta0=copy(theta)
    theta0[0]=0
    
    grad = gradientFunction(theta, X, y) + Lambda * theta0 / m

    return grad