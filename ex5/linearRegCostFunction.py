import numpy as np
def linearRegCostFunction(X, y, theta, Lambda):
    """computes the
    cost of using theta as the parameter for linear regression to fit the
    data points in X and y. Returns the cost in J and the gradient in grad
    """
# Initialize some useful values

    m = y.size # number of training examples
    #X = np.column_stack((np.ones(m), X))

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost and gradient of regularized linear 
#               regression for a particular choice of theta.
#
#               You should set J to the cost and grad to the gradient.
#
    J = ((np.dot(X,theta)-y)**2).sum()/(2*m) + (sum(theta ** 2) - theta[0]**2) * Lambda/(2*m)
    
    theta0=np.copy(theta)
    theta0[0]=0
    
    grad = np.dot(np.dot(X,theta) - y, X)/m + Lambda * theta0 / m

# =========================================================================

    return J, grad