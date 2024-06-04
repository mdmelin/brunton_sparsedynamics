import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


def TVRegDiff(data, iter, alph, u0=None, scale='small', ep=1e-6, dx=None, plotflag=True, diagflag=True):
    """
    TVRegDiff: Total Variation Regularized Differentiation
    
    Parameters:
    data : array_like
        Vector of data to be differentiated.
    iter : int
        Number of iterations to run the main loop.
    alph : float
        Regularization parameter.
    u0 : array_like, optional
        Initialization of the iteration.
    scale : str, optional
        'large' or 'small' (default: 'small').
    ep : float, optional
        Parameter for avoiding division by zero (default: 1e-6).
    dx : float, optional
        Grid spacing (default: 1 / len(data)).
    plotflag : bool, optional
        Flag whether to display plot at each iteration (default: True).
    diagflag : bool, optional
        Flag whether to display diagnostics at each iteration (default: True).
    
    Returns:
    u : array_like
        Estimate of the regularized derivative of data.
    """
    data = np.asarray(data).flatten()
    n = len(data)
    
    if dx is None:
        dx = 1 / n

    if u0 is None:
        u0 = np.concatenate([[0], np.diff(data), [0]])
    
    if scale.lower() == 'small':
        # Construct differentiation matrix.
        c = np.ones(n + 1) / dx
        D = sp.diags([-c, c], [0, 1], shape=(n, n + 1))
        DT = D.T
        # Construct antidifferentiation operator and its adjoint.
        A = lambda x: (np.cumsum(x) - 0.5 * (x + x[0])) * dx
        AT = lambda w: (np.sum(w) * np.ones(n + 1) - np.concatenate([[np.sum(w) / 2], np.cumsum(w) - w / 2])) * dx
        u = u0
        ofst = data[0]
        ATb = AT(ofst - data)
        
        for ii in range(iter):
            Q = sp.diags(1.0 / np.sqrt((D @ u) ** 2 + ep))
            L = dx * DT @ Q @ D
            g = AT(A(u)) + ATb + alph * L @ u
            tol = 1e-4
            maxit = 100
            P = alph * sp.diags(L.diagonal() + 1)
            if diagflag:
                s, _ = spla.cg(lambda v: alph * L @ v + AT(A(v)), g, tol=tol, maxiter=maxit, M=P)
                print(f'iteration {ii+1:4d}: gradient norm = {np.linalg.norm(g):.3e}')
            else:
                s, _ = spla.cg(lambda v: alph * L @ v + AT(A(v)), g, tol=tol, maxiter=maxit, M=P)
            u -= s
            if plotflag:
                plt.plot(u, 'ok')
                plt.draw()
                plt.pause(0.01)
                plt.clf()
    
    elif scale.lower() == 'large':
        A = np.cumsum
        AT = lambda w: np.sum(w) * np.ones(len(w)) - np.concatenate([[0], np.cumsum(w[:-1])])
        c = np.ones(n)
        D = sp.diags([-c, c], [0, 1], shape=(n, n)) / dx
        D[-1, -1] = 0
        DT = D.T
        data -= data[0]
        u = np.concatenate([[0], np.diff(data)])
        ATd = AT(data)
        
        for ii in range(iter):
            Q = sp.diags(1.0 / np.sqrt((D @ u) ** 2 + ep))
            L = DT @ Q @ D
            g = AT(A(u)) - ATd + alph * L @ u
            c = np.cumsum(np.arange(n, 0, -1))
            B = alph * L + sp.diags(c[::-1])
            droptol = 1e-2
            R = spla.spilu(B.tocsc(), drop_tol=droptol)
            M = lambda x: R.solve(R.solve(x, 'T'))
            tol = 1e-4
            maxit = 100
            if diagflag:
                s, _ = spla.cg(lambda x: alph * L @ x + AT(A(x)), -g, tol=tol, maxiter=maxit, M=M)
                print(f'iteration {ii+1:2d}: gradient norm = {np.linalg.norm(g):.3e}')
            else:
                s, _ = spla.cg(lambda x: alph * L @ x + AT(A(x)), -g, tol=tol, maxiter=maxit, M=M)
            u += s
            if plotflag:
                plt.plot(u, 'ok')
                plt.draw()
                plt.pause(0.01)
                plt.clf()
    
    return u

def chop(v):
    return v[1:]

# Example usage:
# data = np.sin(np.linspace(0, 2 * np.pi, 100))
# iter = 100
# alph = 0.1
# u = TVRegDiff(data, iter, alph)