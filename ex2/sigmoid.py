from numpy import e

def sigmoid(z):
    """computes the sigmoid of z."""
    g = 1/(1+e ** (-z))

    return g
