from numpy import log
from sigmoid import sigmoid

def costFunction(theta, X,y):
    """ computes the cost of using theta as the
    parameter for logistic regression and the
    gradient of the cost w.r.t. to the parameters."""
    from numpy import dot
# Initialize some useful values
    m = y.size # number of training examples
    first = -dot(y, log(sigmoid(dot(X,theta))))
    second = -dot((1-y), log(1-sigmoid(dot(X,theta))))
    #first = -dot(y, log(sigmoid(dot(X,theta))))
    #second = -dot((ones(m)-y), log(ones(m)-sigmoid(dot(X,theta))))
    J=(first+second)/m
    return J

