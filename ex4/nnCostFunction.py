import numpy as np

from ex2.sigmoid import sigmoid
#from sigmoidGradient import sigmoidGradient


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):

    """computes the cost and gradient of the neural network. The
  parameters for the neural network are "unrolled" into the vector
  nn_params and need to be converted back into the weight matrices.

  The returned parameter grad should be a "unrolled" vector of the
  partial derivatives of the neural network.
    """

# Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
# for our 2 layer neural network
# Obtain Theta1 and Theta2 back from nn_params
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                       (hidden_layer_size, input_layer_size + 1), order='F').copy()

    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                       (num_labels, (hidden_layer_size + 1)), order='F').copy()



# Setup some useful variables
    m, _ = X.shape


# ====================== YOUR CODE HERE ======================
# Instructions: You should complete the code by working through the
#               following parts.
#
# Part 1: Feedforward the neural network and return the cost in the
#         variable J. After implementing Part 1, you can verify that your
#         cost function computation is correct by verifying the cost
#         computed in ex4.m

    z2 = np.dot(np.column_stack((np.ones((m, 1)), X)), Theta1.T)
    a2 = sigmoid(np.column_stack((np.ones((m, 1)), z2)))
    
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)
    
    nn_hx = a3.ravel()
    #nn_y = np.repeat(y, num_labels)
    nn_y=np.zeros(0)
    for k in range(num_labels):
        nn_y=np.append(nn_y,np.asarray([1 if i==(k+1) else 0 for i in y]))
    MatY=np.reshape(nn_y, (num_labels,m)).T
    
    
    #first = -np.dot(nn_y, np.log(nn_hx))
    #second = -np.dot((1-nn_y), np.log(1-nn_hx))
    first = -np.dot(MatY.T, np.log(a3))
    second = -np.dot((1-MatY.T), np.log(1-a3))
    J=(first+second)/m

#
# Part 2: Implement the backpropagation algorithm to compute the gradients
#         Theta1_grad and Theta2_grad. You should return the partial derivatives of
#         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
#         Theta2_grad, respectively. After implementing Part 2, you can check
#         that your implementation is correct by running checkNNGradients
#
#         Note: The vector y passed into the function is a vector of labels
#               containing values from 1..K. You need to map this vector into a 
#               binary vector of 1's and 0's to be used with the neural network
#               cost function.
#
#         Hint: We recommend implementing backpropagation using a for-loop
#               over the training examples if you are implementing it for the 
#               first time.
#
# Part 3: Implement regularization with the cost function and gradients.
#
#         Hint: You can implement this around the code for
#               backpropagation. That is, you can compute the gradients for
#               the regularization separately and then add them to Theta1_grad
#               and Theta2_grad from Part 2.
#



    # -------------------------------------------------------------

    # =========================================================================

    
    # Unroll gradient
    #grad = np.hstack((Theta1_grad.T.ravel(), Theta2_grad.T.ravel()))
    grad = 0

    return J, grad