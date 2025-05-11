""" 
###############################################################################

        Credit to Enrico Zorzetto, 9/10/2017
        enrico.zorzetto@duke.edu
        
###############################################################################
"""

import numpy as np
import scipy as sc
from scipy.special import gamma
from scipy.stats import exponweib


###############################################################################
###############################################################################

########################### WEIBULL DISTRIBUTION ##############################

###############################################################################
###############################################################################


def wei_fit(sample, how="pwm", threshold=0, std=False, std_how="boot", std_num=1000):
    """fit Weibull with one of the following methods:
    -----------------------------------------------------------------------
    how = 'pwm' for probability weighted moments
    how = 'ml' for maximum likelihood
    how = 'ls' for least squares
    -----------------------------------------------------------------------
    choice of threshold available for PWM only
    without renormalization (default is zero threshold)
    -----------------------------------------------------------------------
    optional: if std = True (default is false)
    compute parameter est. standard deviations. parstd
    and their covariance matrix varcov
    if std_how = 'boot' bootstrap is used
    if std_how = 'hess' hessian is used (only available for max like.)
    std_num --> number or resamplings in the bootstrap procedure.
    default is 1000.
    --------------------------------------------------------------------"""
    # print('how = ', how)
    if np.isnan(sample).all():
        return np.nan, np.nan, np.nan

    if how == "pwm":
        N, C, W = wei_fit_pwm(sample, threshold=threshold)
    elif how == "ml":
        N, C, W = wei_fit_ml(sample)
    elif how == "ls":
        N, C, W = wei_fit_ls(sample)
    else:
        print(" ERROR - insert a valid fitting method ")
    parhat = N, C, W
    if std == True:
        if how == "pwm":
            parstd, varcov = wei_boot(
                sample, fitfun=wei_fit_pwm, npar=2, ntimes=std_num
            )
        elif how == "ls":
            parstd, varcov = wei_boot(sample, fitfun=wei_fit_ls, npar=2, ntimes=std_num)
        elif how == "ml" and std_how == "boot":
            parstd, varcov = wei_boot(sample, fitfun=wei_fit_ml, npar=2, ntimes=std_num)
        elif how == "ml" and std_how == "hess":
            print(" wei_fit ERROR: 'hess' CIs not available yet")
            ni, ci, wi, parstd, varcov = wei_fit_ml(sample, std=True)
        else:
            print("wei_fit ERROR: insert a valid method for CIs")
        return parhat, parstd, varcov
    else:
        return parhat


def wei_boot(sample, fitfun, npar=2, ntimes=1000):
    """non parametric bootstrap technique
    for computing confidence interval for a distribution
    (when I do not know the asymptotic properties of the distr.)
    return std and optional pdf of fitted parameters
    and their covariance matrix varcov
    fit to a sample of a distribution using the fitting function fitfun
    with a number of parameters npar
    ONLY FOR WEIBULL
    Ignore the first output parameter - N"""
    n = np.size(sample)
    # resample from the data with replacement
    parhats = np.zeros((ntimes, npar))
    for ii in range(ntimes):
        replaced = np.random.choice(sample, n)
        NCW = fitfun(replaced)
        parhats[ii, :] = NCW[1:]
    parstd = np.std(parhats, axis=0)
    varcov = np.cov(parhats, rowvar=False)
    return parstd, varcov


def wei_fit_pwm(sample, threshold=0):
    """fit a 2-parameters Weibull distribution to a sample
    by means of Probability Weighted Moments (PWM) matching (Greenwood 1979)
    using only observations larger than a value 'threshold' are used for the fit
    -- threshold without renormalization -- it assumes the values below are
    not present. Default threshold = 0
    INPUT:: sample (array with observations)
           threshold (default is = 0)
    OUTPUT::
    returns dimension of the sample (n) (only values above threshold)
    Weibull scale (c) and shape (w) parameters"""
    sample = np.asarray(sample)  # from list to Numpy array
    wets = sample[sample > threshold]
    x = np.sort(wets)  # sort ascend by default
    M0hat = np.mean(x)
    M1hat = 0.0
    n = x.size  # sample size
    for ii in range(n):
        real_ii = ii + 1
        M1hat = M1hat + x[ii] * (n - real_ii)
    M1hat = M1hat / (n * (n - 1))
    c = M0hat / gamma(np.log(M0hat / M1hat) / np.log(2))  # scale par
    w = np.log(2) / np.log(M0hat / (2 * M1hat))  # shape par
    return n, c, w


def wei_fit_pwm_cens(sample, threshold=0):
    """fit a 2-parameters Weibull distribution to a sample
    by means of censored Probability Weighted Moments (CPWM) - Wang, 1999
    only observations larger than a value 'threshold' are used for the fit
    but the probability mass of the observations below threshold is accounted for.
    compute the first two PWMs
    ar and br are linear comb of each other, perfectly equivalent
    I use censoring on the br as proposed by Wang 1990
    so that I am censoring the lower part of the distribution
    Default threshold = 0
    INPUT:: sample (array with observations)
           threshold (default is = 0)
    OUTPUT::
    returns numerosity of the sample (n) (only values above threshold)
    Weibull scale (c) and shape (w) parameters"""
    sample = np.asarray(sample)  # from list to Numpy array
    wets = sample[sample > 0]
    x = np.sort(wets)  # sort ascend by default
    b0 = 0.0
    b1 = 0.0
    n = x.size  # sample size
    for ii in range(n):
        real_ii = ii + 1
        if x[ii] > threshold:
            b1 = b1 + x[ii] * (real_ii - 1)
            b0 = b0 + x[ii]
    b1 = b1 / (n * (n - 1))
    b0 = b0 / n
    # obtain ar=Mrhat  as linear combination of the first two br
    M0hat = b0
    M1hat = b0 - b1
    c = M0hat / gamma(np.log(M0hat / M1hat) / np.log(2))  # scale par
    w = np.log(2) / np.log(M0hat / (2 * M1hat))  # shape par
    return n, c, w


def wei_quant(Fi, C, w, ci=False, varcov=[]):
    """WEI quantiles and (optional) confidence intervals"""
    Fi = np.asarray(Fi)
    is_scalar = False if Fi.ndim > 0 else True
    Fi.shape = (1,) * (1 - Fi.ndim) + Fi.shape
    q = (-np.log(1 - Fi)) ** (1 / w) * C
    q = q if not is_scalar else q[0]
    if ci == True:
        # compute std of quantiles using the DELTA METHOD
        m = np.size(Fi)
        qu = np.zeros(m)
        ql = np.zeros(m)
        for ii in range(m):
            yr = 1 - Fi[ii]
            # dx/dC and dx/dw
            DEL = np.array(
                [
                    (-np.log(yr)) ** (1 / w),
                    C * (-np.log(yr)) ** (1 / w) * np.log(-np.log(1 - Fi[ii])),
                ]
            )
            prod1 = np.dot(varcov, DEL)
            varz = np.dot(prod1, DEL)
            stdz = np.sqrt(varz)
            ql[ii] = q[ii] - 1.96 * stdz
            qu[ii] = q[ii] + 1.96 * stdz
        qu = qu if not is_scalar else qu[0]
        ql = ql if not is_scalar else ql[0]
        return q, qu, ql
    else:
        return q


def wei_pdf(x, C, W):
    """compute Weibull pdf with parameters scale C and shape w
    for a scalar OR array input of positive values x"""
    x = np.asarray(x)  # transform to numpy array
    is_scalar = False if x.ndim > 0 else True  # create flag for output
    x.shape = (1,) * (1 - x.ndim) + x.shape  # give it dimension 1 if scalar
    pdf = W / C * (x / C) ** (W - 1) * np.exp(-((x / C) ** W))
    pdf = pdf if not is_scalar else pdf[0]
    return pdf


def wei_mean_variance(C, w):
    """Computes mean mu and variance var
    of a Weibull distribution with parameter scale C and shape w
    -or repeat for all the elements for same-dim arrays C and W
    NOTE: C and w need to have the same dimension e data type"""
    C = np.asarray(C)
    w = np.asarray(w)
    # if C is scalar, we return both scalars
    is_C_scalar = False if C.ndim > 0 else True
    C.shape = (1,) * (1 - C.ndim) + C.shape
    w.shape = (1,) * (1 - w.ndim) + w.shape
    # compute mean and variance
    mu = C / w * gamma(1 / w)
    var = C**2 / w**2 * (2 * w * gamma(2 / w) - (gamma(1 / w)) ** 2)
    mu = mu if not is_C_scalar else mu[0]
    var = var if not is_C_scalar else var[0]
    return mu, var


def wei_cdf(q, C, w):  # modified order
    """returns the non exceedance probability of quantiles q (scalar or array)
    for a Weibull distribution with shape w and scale C"""
    q = np.asarray(q)
    is_scalar = False if q.ndim > 0 else True
    cdf = 1 - np.exp(-((q / C) ** w))
    cdf = cdf if not is_scalar else cdf[0]
    return cdf


def wei_surv(q, C, w):  # modified order
    """returns the survival probability of quantiles q (scalar or array)
    for a Weibull distribution with shape w and scale C"""
    q = np.asarray(q)
    is_scalar = False if q.ndim > 0 else True
    sdf = np.exp(-((q / C) ** w))
    sdf = sdf if not is_scalar else sdf[0]
    return sdf


def wei_random_quant(length, C, w):
    """generates a vector of length 'length' of
    quantiles randomly extracted from a Weibull distr with par C, w
    if length = 1, returns a scalar"""
    Fi = np.random.rand(length)
    xi = (-np.log(1 - Fi)) ** (1 / w) * C
    xi = xi if length > 1 else xi[0]
    return xi


def wei_fit_ls(sample):
    """
    fit Weibull distribution to given sample
    removing data that are not positive
    return N (number obs >0)
    C (scale par) and w (shape par)
    """
    sample = np.array(sample)
    sample2 = sample[sample > 0]
    xi = np.sort(sample2)
    N = len(xi)
    II = np.arange(1, N + 1)
    Fi = II / (N + 1)
    yr = np.log(-np.log(1 - Fi))
    xr = np.log(xi)

    xrbar = np.mean(xr)
    yrbar = np.mean(yr)
    xrstd = np.std(xr)
    yrstd = np.std(yr)

    w = yrstd / xrstd  # shape par
    C = np.exp(xrbar - 1 / w * yrbar)  # scale par
    return N, C, w


def wei_fit_mlpy(sample):
    # remove in a future version. the other ml fit works fine
    """
    fit Weibull using the builtin scipy exponweib function
    setting floc = 0 and fa = 1
    return n size of sample >0 (used for fit)
    """
    sample = np.array(sample)
    sample2 = sample[sample > 0]
    n = len(sample2)
    # setting a = 1 (additional generalized exponent)
    # and floc = 0 (zero location here)
    a, w, mu, c = exponweib.fit(sample2, floc=0, fa=1)
    # do not return 2nd shape par. a and loc mu
    return n, c, w


def wei_fit_ml(sample, std=False):
    """--------------------------------------------------------------
    fit Weibull by means of Maximum_Likelihood _Estimator (MLE)
    finding numerically the max of the likelihood function
    return n size of sample >0 (used for fit)
    if std = True compute standard deviations and covariances
    of parameters C and w.
    ----------------------------------------------------------------"""
    from numpy.linalg import inv

    def hess(fun, y, data):
        """numeric Hessian matrix
        for estimating MLE parameters confidence intervals"""
        ep = 0.0001
        x = np.array(y)
        eps = ep * x
        n = np.size(x)
        m = np.zeros((n, n))
        x1 = np.zeros(n)
        x2 = np.zeros(n)
        x3 = np.zeros(n)
        x4 = np.zeros(n)
        for i in range(n):
            for j in range(n):
                x1[:] = x[:]
                # I modify the original array as well - it is a view!
                x1[i] = x1[i] + eps[i]
                x1[j] = x1[j] + eps[j]
                x2[:] = x[:]
                x2[i] = x2[i] + eps[i]
                x2[j] = x2[j] - eps[j]
                x3[:] = x[:]
                x3[i] = x3[i] - eps[i]
                x3[j] = x3[j] + eps[j]
                x4[:] = x[:]
                x4[i] = x4[i] - eps[i]
                x4[j] = x4[j] - eps[j]
                m[i, j] = (
                    fun(x1, data) - fun(x2, data) - fun(x3, data) + fun(x4, data)
                ) / (4 * eps[i] * eps[j])
        M = np.asmatrix(m)
        return inv(M)

    sample = np.array(sample)
    sample2 = sample[sample > 0]
    x = sample2
    n = len(x)
    # derivative of the log likelihood function with respect with par. w:
    like = lambda w: n * (
        1 / w - np.sum((x**w) * np.log(x)) / np.sum(x**w)
    ) + np.sum(np.log(x))
    w_init_guess = 1.0
    w_hat = sc.optimize.fsolve(like, w_init_guess)[0]
    c_hat = (np.sum(x**w_hat) / n) ** (1.0 / w_hat)
    parhat = (c_hat, w_hat)
    if std:
        varcov = hess(wei_negloglike, parhat, sample)
        parstd = np.sqrt(np.diag(varcov))
        return n, c_hat, w_hat, parstd, varcov
    else:
        return n, c_hat, w_hat


def wei_negloglike(parhat, data):
    """compute Weibull neg log likelihood function
    for a given sample xi and estimated parameters C,w"""
    C = parhat[0]
    w = parhat[1]
    xi = data[data > 0]
    N = len(xi)
    nllw = -N * np.log(w / C) - (w - 1) * np.sum(np.log(xi / C)) + np.sum((xi / C) ** w)
    return nllw
