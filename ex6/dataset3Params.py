import numpy as np
import sklearn.svm


def dataset3Params(X, y, Xval, yval):
    """returns your choice of C and sigma. You should complete
    this function to return the optimal C and sigma based on a
    cross-validation set.
    """

# You need to return the following variables correctly.



    values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    best_precision = 0
    
    for c_value, sigma_value in [(a,b) for a in values for b in values]:
        #print c_value, sigma_value
        gamma = 1.0 / (2.0 * sigma_value ** 2)
        # Train the SVM
        clf = sklearn.svm.SVC(C=c_value, kernel='rbf', tol=1e-3, max_iter=200, gamma=gamma)
        clf.fit(X, y)
        predictions = clf.predict(Xval)
        # Evaluate the model
        precision = np.mean(predictions == yval)
        if precision > best_precision:
            C = c_value
            sigma = sigma_value
            best_precision = precision
    #print C, sigma
        
# ====================== YOUR CODE HERE ======================
# Instructions: Fill in this function to return the optimal C and sigma
#               learning parameters found using the cross validation set.
#               You can use svmPredict to predict the labels on the cross
#               validation set. For example, 
#                   predictions = svmPredict(model, Xval)
#               will return the predictions on the cross validation set.
#
#  Note: You can compute the prediction error using 
#        mean(double(predictions ~= yval))
#


# =========================================================================
    return C, sigma
