"""
Non Negative Matrix Factorization with projected gradient
Created on 03/2019

@author: E.L. Benaroya - laurent.benaroya@gmail.com

Copyright 2018 E.L. Benaroya
This software is distributed under the terms of the GNU Public License
version 3 (http://www.gnu.org/licenses/gpl.txt)

"""
import numpy as np
import itertools
from nmf.alpha_rnd import alpha_rnd

spinner = itertools.cycle(['\\', '|', '/', '-'])

# cost function and gradient


def beta_func(X, v, ftype='fro'):
    """
    compute cost function
    Parameters
    ----------
    X : 2D numpy array
        input non negative data (spectrogram)
    v : 2D numpy array
        non negative approximation of X
    ftype : string
        type of nmf cost (default 'fro')
        'fro' : Frobenius square norm
        'kl' : Kullback-Leibler
        'is' : Itakura - Saito
    Returns
    -------
    cost : float
        cost associated with the cost function type
    """
    if ftype.lower() == 'fro':
        cost = 0.5*np.sum(np.abs(X - v) ** 2)
    elif ftype.lower() == 'kl':
        # Kullback-Liebler
        cost = np.sum(X * np.log(X / v) + v - X)
    elif ftype.lower() == 'is':
        cost = np.sum(np.log(v) + X / v)
    else:
        raise Exception('Unknown cost function')
    return cost


def beta_grad(X, v, ftype='fro'):
    """
    compute gradient of the cost function with respect to v

    Parameters
    ----------
    X : 2D numpy array
        input non negative data (spectrogram)
    v : 2D numpy array
        non negative approximation of X
    ftype : string
        type of nmf cost (default 'fro')
    Returns
    -------
    grad : 2D numpy array
        gradient
    """
    if ftype.lower() == 'fro':
        grad = v - X
    elif ftype.lower() == 'kl':
        grad = -X / v + 1
    elif ftype.lower() == 'is':
        grad = 1 / v - X / (v ** 2)
    else:
        raise Exception('Unknown cost function')

    return grad


def decr(it):
    # pow = 2
    return (it+1)  # *np.log(it+1)  # np.power(it, pow)


def projgradnmf(X, W=1, indW=None, H=None, Beta=2, nbIter=100, noiseFloor=0.,
                minVal=1e-16, Wupdate=True, Hupdate=True,
                use_alpha_stable=False, Alpha=1.8, sigma_alpha=1e-2):
    """
    projeted gradient based NMf with Beta-divergence

    Parameters
    ----------
    X : array, shape (F, N) - F = frequency bins, N = number of frames
        Matrix to factorize

    W : int or array, shape (F, K) - if W is an integer, then W is randomly drawn with K = W components
        template matrix or number of templates

    indW : vector (K1,) - indices of the dictionary components that will be updated

    H : array, shape(K,N) - activation matrix (optional)

    Beta : float, Beta-divergence parameter (default = 0., Itakura-Saito divergence)

    nbIter : int, number of iterations (default = 100)

    noiseFloor : float, added value to the data X and its approximation V to avoid numerical problems, especially with
    Itakura-Saito divergence

    minVal : float, minimum value - sort of equivalent to 'eps' in matlab :(

    Wupdate : bool, update the template matrix W if True (default = True)

    Hupdate : bool, update the activation matrix H if True (default = True)

    use_alpha_stable  : bool
        use alpha-stable noise instead of noiseFloor

    Alpha : float
        alpha parameter for alpha-stable noise

    sigma_alpha : float
        intensity of the "injected" noise

    Returns
    -------
    W : array, shape(F,K) - template matrix

    H : array, shape(K,N) - activation matrix

    V : array, shape(F,N) - approximation of X, V = np.dot(W,H)+noiseFloor

    Cost : vector, shape(nbIter,) - total cost at each iteration = Beta divergence D(X,V) + constraint cost C(H)

    mu_mem : array (nbIter, 2)
        value of step size for W and H across iterations
    """
    F, N = X.shape
    if isinstance(W, int):
        K = W
        W = np.random.rand(F, K)+minVal
        indW = range(K)
    else:
        K = W.shape[1]

    if indW is None:
        indW = range(K)

    if H is None:
        H = np.random.rand(K, N)+minVal
    indW
    Kw = len(indW)
    X = X+noiseFloor
    Cost = np.zeros((nbIter,))

    # min W and H values
    w_min = 1e-10
    h_min = 1e-10

    # Wolfe constants
    epsilon1 = 0.0001
    epsilon2 = 0.9

    # COPY W and H!
    wk = np.copy(W)
    hk = np.copy(H)

    if Beta == 2:
        cost_type = 'fro'
    elif Beta == 1:
        cost_type = 'kl'
    elif Beta == 0:
        cost_type = 'is'
    else:
        raise Exception('Wrong Beta value')

    print("Cost type %s" % cost_type)

    dwk_old = 0.

    print("Number of iterations %d" % nbIter)
    Cost = np.zeros((nbIter,))

    # line search scale parameters
    mu_mem = np.zeros((nbIter, 3))

    # for stopping criterion
    fxk_old = 0.
    wk_old = np.zeros_like(wk)
    hk_old = np.zeros_like(hk)

    # Wolfe-Armijo line search, see :
    # https://sites.math.washington.edu/~burke/crs/408/notes/nlp/line.pdf
    for it in range(1, nbIter + 1):

        v = wk.dot(hk) + noiseFloor
        fk = beta_func(X, v, cost_type)
        Cost[it - 1] = fk
        fxk = fk  # for stopping criterion

        # #######################
        # compute gradient on H @
        # ##################### #
        if Hupdate:
            v = wk.dot(hk) + noiseFloor
            fk = beta_func(X, v, cost_type)

            dfk = beta_grad(X, v, cost_type)
            dhk = wk.T.dot(dfk)
            if use_alpha_stable:
                alpha_noise = sigma_alpha / decr(it) * alpha_rnd(Alpha, K, 1)
                dhk += alpha_noise

            # init line search
            m = -np.sum(dhk ** 2)
            a = 0.
            b = + np.inf
            t = .1

            # Wolfe conditions line search
            for line_ls in range(100):
                hk_ls = np.maximum(hk - t * dhk, h_min)
                v_ls = wk.dot(hk_ls) + noiseFloor

                fk_ls = beta_func(X, v_ls, cost_type)

                if fk_ls > fk + t * epsilon1 * m:
                    b = t
                    t = 0.5 * (a + b)
                else:
                    dfk_ls = beta_grad(X, v_ls, cost_type)
                    dhk_ls = wk.T.dot(dfk_ls)
                    if use_alpha_stable:
                        dhk_ls += alpha_noise
                    if -np.sum(dhk_ls * dhk) < epsilon2 * m:
                        a = t
                        if b == +np.inf:
                            t = 2 * a
                        else:
                            t = 0.5 * (a + b)
                    else:
                        break
            hk = hk_ls
            mu_mem[it - 1, 1] = t

        # ##################### #
        # compute gradient on W #
        # ##################### #
        if Wupdate:
            v = wk.dot(hk) + noiseFloor

            fk = beta_func(X, v, cost_type)
            dfk = beta_grad(X, v, cost_type)
            hkw = hk[indW, :]
            dwk = dfk.dot(hkw.T)
            if use_alpha_stable:
                # alpha_noise = noiseFloor  # sigma_alpha / decr(it) * alpha_rnd(Alpha, 1, N)
                alpha_noise = sigma_alpha / decr(it) * alpha_rnd(Alpha, 1, len(indW))
                dwk += alpha_noise

            m = -np.sum(dwk ** 2)
            a = 0.
            b = + np.inf
            t = .1

            for line_ls in range(100):
                wk_ls = np.maximum(wk[:, indW] - t * dwk, w_min)

                v_ls = v + (- wk[:, indW] + wk_ls).dot(hkw)

                fk_ls = beta_func(X, v_ls, cost_type)

                if fk_ls > fk + t * epsilon1 * m:
                    b = t
                    t = 0.5 * (a + b)
                else:
                    dfk_ls = beta_grad(X, v_ls, cost_type)
                    dwk_ls = dfk_ls.dot(hkw.T)
                    if use_alpha_stable:
                        dwk_ls += alpha_noise
                    if -np.sum(dwk_ls * dwk) < epsilon2 * m:
                        a = t
                        if b == +np.inf:
                            t = 2 * a
                        else:
                            t = 0.5 * (a + b)
                    else:
                        break

            wk[:, indW] = wk_ls
            mu_mem[it - 1, 0] = t
        mu_mem[it-1, 2] = sigma_alpha / decr(it)
        if it % 10 == 0:
            print('%s iter %d, mu w %e, alpha h %e, cost %e\r' %
                  (next(spinner), it, mu_mem[it - 1, 0], mu_mem[it - 1, 1], fk), end='')
        # stopping criteria
        epsilon = 1e-6
        min_iterations = 100

        # variation of the parameters criterion
        normw2 = np.linalg.norm(wk - wk_old, 'fro') ** 2
        normw02 = np.linalg.norm(wk_old, 'fro') ** 2
        normh2 = np.linalg.norm(hk - hk_old, 'fro') ** 2
        normh02 = np.linalg.norm(hk_old, 'fro') ** 2

        criterion_z = np.sqrt(normw2 + normh2) / (1 + np.sqrt(normw02 + normh02)) < epsilon

        # variation of the function criterion
        criterion_f = np.abs(fxk - fxk_old) / (1 + np.abs(fxk_old)) < epsilon

        # TODO : add a criterion on the gradient? (norm(gradient) < epsilon???

        if it > min_iterations and criterion_f and criterion_z:
            print('\n')
            print('stopping criterion met after %d iterations' % it)
            Cost = Cost[:it]
            mu_mem = mu_mem[:it, :]
            break
        fxk_old = fxk
        wk_old = wk
        hk_old = hk

    v = wk.dot(hk) + noiseFloor
    return wk, hk, v, Cost, mu_mem
