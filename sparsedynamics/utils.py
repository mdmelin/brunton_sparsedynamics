import numpy as np
from numba import jit

def sparsifyDynamics(Theta, dXdt, lambda_val, n):
    """
    Copyright 2015, All Rights Reserved
    Code by Steven L. Brunton
    For Paper, "Discovering Governing Equations from Data: 
           Sparse Identification of Nonlinear Dynamical Systems"
    by S. L. Brunton, J. L. Proctor, and J. N. Kutz

    Python port by Max MElin, 2024

    Parameters:
    Theta : ndarray
        Library of candidate functions
    dXdt : ndarray
        Time derivatives of the state
    lambda_val : float
        Sparsification knob
    n : int
        State dimension

    Returns:
    Xi : ndarray
        Sparse coefficients matrix
    """
    # Compute Sparse regression: sequential least squares
    Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0]  # initial guess: Least-squares

    # Lambda is our sparsification knob
    for k in range(10):
        smallinds = np.abs(Xi) < lambda_val   # find small coefficients
        Xi[smallinds] = 0                     # and threshold
        for ind in range(n):                  # n is state dimension
            biginds = ~smallinds[:, ind]
            # Regress dynamics onto remaining terms to find sparse Xi
            Xi[biginds, ind] = np.linalg.lstsq(Theta[:, biginds], dXdt[:, ind], rcond=None)[0]
    
    return Xi

@jit(nopython=True)
def poolData(yin, nVars, polyorder, usesine):
    """
    Copyright 2015, All Rights Reserved
    Code by Steven L. Brunton
    For Paper, "Discovering Governing Equations from Data: 
           Sparse Identification of Nonlinear Dynamical Systems"
    by S. L. Brunton, J. L. Proctor, and J. N. Kutz
    
    Python port by Max Melin, 2024

    Parameters:
    yin : ndarray
        Input data, shape (n_samples, nVars)
    nVars : int
        Number of variables
    polyorder : int
        Polynomial order
    usesine : bool
        Flag to include sine and cosine functions

    Returns:
    yout : ndarray
        Output data with polynomial and trigonometric terms
    """
    n = yin.shape[0]
    ind = 0

    # Initialize yout with the correct size
    yout = np.ones((n, 1))
    ind += 1

    # poly order 1
    for i in range(nVars):
        yout = np.hstack((yout, yin[:, i:i+1]))
        ind += 1

    if polyorder >= 2:
        # poly order 2
        for i in range(nVars):
            for j in range(i, nVars):
                yout = np.hstack((yout, (yin[:, i] * yin[:, j]).reshape(n, 1)))
                ind += 1

    if polyorder >= 3:
        # poly order 3
        for i in range(nVars):
            for j in range(i, nVars):
                for k in range(j, nVars):
                    yout = np.hstack((yout, (yin[:, i] * yin[:, j] * yin[:, k]).reshape(n, 1)))
                    ind += 1

    if polyorder >= 4:
        # poly order 4
        for i in range(nVars):
            for j in range(i, nVars):
                for k in range(j, nVars):
                    for l in range(k, nVars):
                        yout = np.hstack((yout, (yin[:, i] * yin[:, j] * yin[:, k] * yin[:, l]).reshape(n, 1)))
                        ind += 1

    if polyorder >= 5:
        # poly order 5
        for i in range(nVars):
            for j in range(i, nVars):
                for k in range(j, nVars):
                    for l in range(k, nVars):
                        for m in range(l, nVars):
                            yout = np.hstack((yout, (yin[:, i] * yin[:, j] * yin[:, k] * yin[:, l] * yin[:, m]).reshape(n, 1)))
                            ind += 1

    if usesine:
        for k in range(1, 11):
            yout = np.hstack((yout, np.sin(k * yin), np.cos(k * yin)))

    return yout

    
#def poolDataLIST(yin, ahat, nVars, polyorder, usesine):
#    """
#    Copyright 2015, All Rights Reserved
#    Code by Steven L. Brunton
#    For Paper, "Discovering Governing Equations from Data: 
#           Sparse Identification of Nonlinear Dynamical Systems"
#    by S. L. Brunton, J. L. Proctor, and J. N. Kutz
#
#    Parameters:
#    yin : list of str
#        Input variables names
#    ahat : ndarray
#        Coefficients array
#    nVars : int
#        Number of variables
#    polyorder : int
#        Polynomial order
#    usesine : bool
#        Flag to include sine and cosine functions
#
#    Returns:
#    newout : list of lists
#        Output list with terms and coefficients
#    """
#    n = len(yin)
#    yout = []
#
#    ind = 0
#    # poly order 0
#    yout.append(['1'])
#    ind += 1
#
#    # poly order 1
#    for i in range(nVars):
#        yout.append([yin[i]])
#        ind += 1
#
#    if polyorder >= 2:
#        # poly order 2
#        for i in range(nVars):
#            for j in range(i, nVars):
#                yout.append([yin[i] + yin[j]])
#                ind += 1
#
#    if polyorder >= 3:
#        # poly order 3
#        for i in range(nVars):
#            for j in range(i, nVars):
#                for k in range(j, nVars):
#                    yout.append([yin[i] + yin[j] + yin[k]])
#                    ind += 1
#
#    if polyorder >= 4:
#        # poly order 4
#        for i in range(nVars):
#            for j in range(i, nVars):
#                for k in range(j, nVars):
#                    for l in range(k, nVars):
#                        yout.append([yin[i] + yin[j] + yin[k] + yin[l]])
#                        ind += 1
#
#    if polyorder >= 5:
#        # poly order 5
#        for i in range(nVars):
#            for j in range(i, nVars):
#                for k in range(j, nVars):
#                    for l in range(k, nVars):
#                        for m in range(l, nVars):
#                            yout.append([yin[i] + yin[j] + yin[k] + yin[l] + yin[m]])
#                            ind += 1
#
#    if usesine:
#        for k in range(1, 11):
#            yout.append(['sin(' + str(k) + '*yin)'])
#            ind += 1
#            yout.append(['cos(' + str(k) + '*yin)'])
#            ind += 1
#
#    output = yout
#    newout = [[''] * (1 + nVars)]
#    for k in range(n):
#        newout[0][1 + k] = yin[k] + 'dot'
#    
#    for k in range(len(ahat)):
#        newout.append([output[k][0]] + list(ahat[k]))
#
#    return newout
#