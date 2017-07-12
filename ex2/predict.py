from numpy import round, dot

from sigmoid import sigmoid


def predict(theta, X):

    """ computes the predictions for X using a threshold at 0.5
    (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    """

    p = round(sigmoid(dot(X,theta)))

    return p