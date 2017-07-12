from sigmoid import sigmoid
from numpy import squeeze, asarray, dot

def gradientFunction(theta, X, y):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """

    m = len(y)   # number of training examples

    grad = dot(sigmoid(dot(X,theta.T)) - y, X)/m

    return grad
