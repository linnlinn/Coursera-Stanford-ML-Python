import numpy as np

from ex2.sigmoid import sigmoid

def predictOneVsAll(all_theta, X):
    """will return a vector of predictions
  for each example in the matrix X. Note that X contains the examples in
  rows. all_theta is a matrix where the i-th row is a trained logistic
  regression theta vector for the i-th class. You should set p to a vector
  of values from 1..K (e.g., p = [1 3 1 2] predicts classes 1, 3, 1, 2
  for 4 examples) """

    m = X.shape[0]

    # You need to return the following variables correctly
    p = np.zeros((m, 1))

    # Add ones to the X data matrix
    X = np.column_stack((np.ones((m, 1)), X))

    p = np.argmax(np.dot(X,all_theta.T),axis=1)
    return p + 1    # add 1 to offset index of maximum in A row
