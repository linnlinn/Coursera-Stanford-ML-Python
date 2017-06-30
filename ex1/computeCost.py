import numpy as np

def computeCost(X, y, theta):
    """
       computes the cost of using theta as the parameter for linear 
       regression to fit the data points in X and y
    """
    m = y.size
    #J =  sum((np.apply_along_axis(sum,1,theta*X) - y) **2) /(2*m)
    J =  np.linalg.norm(np.apply_along_axis(sum,1,theta*X) - y)**2/(2*m)

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.


# =========================================================================

    return J


