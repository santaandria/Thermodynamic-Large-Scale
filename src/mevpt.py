###########################
# Author: Santa Andria #
###########################
import pandas as pd
import numpy as np
from scipy.special import gamma
from scipy.optimize import minimize, fsolve, brentq, minimize_scalar
from scipy.stats import linregress
import warnings
from functools import partial
from tqdm import tqdm


def wei_fit(sample, threshold=0, ci=False, ntimes=1000):
    #### Modified from Enrico Zorzetto/Mevpy
    if np.isnan(sample).all():
        return np.nan, np.nan, np.nan

    if ci:
        mean, ci_low, ci_up = wei_boot_ci(
            sample, fitfun=wei_fit_pwm, confidence_level=0.95, ntimes=ntimes
        )
        return (*mean, *ci_low, *ci_up)  # c_hat, w_hat, c_low, w_low, c_up, w_up
    else:
        N, C, W = wei_fit_pwm(sample, threshold=threshold)
        return (N, C, W)


def wei_boot_ci(sample, fitfun, confidence_level=0.95, ntimes=1000):
    #### Modified from Enrico Zorzetto/Mevpy
    n = np.size(sample)
    # resample from the data with replacement
    parhats = np.zeros((ntimes, 2))

    alpha = 1 - confidence_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)

    for ii in range(ntimes):
        replaced = np.random.choice(sample, n, replace=True)
        NCW = fitfun(replaced)
        parhats[ii, :] = NCW[1:]
    ci_low = np.percentile(parhats, lower_percentile, axis=0)
    ci_up = np.percentile(parhats, upper_percentile, axis=0)
    mean = np.mean(parhats, axis=0)
    return mean, ci_low, ci_up


def wei_fit_pwm(sample, threshold=0):
    #### From Enrico Zorzetto/Mevpy
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


def bin_ps_data(prcp, exg, Nbins):
    """
    Bin P-S data into bins of equal size
    Args:
    -----
    prcp: Precipitation events data without NaNs
    exg: Exogenous variable data (e.g. TMEAN) without NaNs, corresponding to the precipitation events
    """
    # dim = min(len(prcp), len(exg))
    dim = len(prcp)  # len(prcp) = len(exg)
    binsize = int(dim / Nbins)

    prcp = prcp[: binsize * Nbins]  # Truncate to fit into bins
    exg = exg[: binsize * Nbins]

    # Bin data (# Each row corresponds to each bin)
    alloc_vec = np.argsort(exg, kind="stable")
    exg_mat = exg[alloc_vec].reshape((Nbins, binsize))
    prcp_mat = prcp[alloc_vec].reshape((Nbins, binsize))
    return prcp_mat, exg_mat


def pwm_estimation(X, threshold=1, ci=False, ntimes=1000):
    """
    Perform Parameter Estimation using PWM (Probability-Weighted Moments) method.

    Args:
        X (NumPy 2D array): Events matrix where each row corresponds to a bin
        thresh: Independent events > threshold are used to fit data

    Returns:
        DataFrame: with PWM parameters 'c', 'w', 'T_med'
    """
    X = X - threshold

    pwm_params = np.apply_along_axis(wei_fit, axis=1, arr=X, ci=ci, ntimes=ntimes)
    return pwm_params


def scaling_analysis(prcp, exg, var, ci=False, ntimes=1000, n_bins=20):
    threshold = np.percentile(prcp, 10)
    prcp_mat, exg_mat = bin_ps_data(prcp, exg, Nbins=n_bins)

    # Fit Weibull Distribution at at each bin
    pwm_params = pwm_estimation(prcp_mat, threshold=threshold, ci=ci, ntimes=ntimes)
    exg_med = np.median(exg_mat, axis=1)

    if ci:
        cw_scaling = pd.DataFrame(
            {
                "c": pwm_params[:, 0],
                "w": pwm_params[:, 1],
                "c_low": pwm_params[:, 2],
                "w_low": pwm_params[:, 3],
                "c_up": pwm_params[:, 4],
                "w_up": pwm_params[:, 5],
                f"{var}_med": exg_med,
            }
        )
    else:
        cw_scaling = pd.DataFrame(
            {
                "c": pwm_params[:, 1],
                "w": pwm_params[:, 2],
                f"{var}_med": exg_med,
            }
        )

    if var == "TMEAN" or var == "DPT":
        cw_scaling["c"] = np.log10(cw_scaling["c"])
        if ci:
            cw_scaling["c_up"] = np.log10(cw_scaling["c_up"])
            cw_scaling["c_low"] = np.log10(cw_scaling["c_low"])

    return cw_scaling


def load_data(stn_id, d, data_dir):
    df = pd.read_csv(
        data_dir / d / (stn_id + ".csv"), index_col=0, parse_dates=True
    ).dropna()

    mask = (
        (df["PRCP"] >= 0.254)
        if "DPT" in df.columns
        else (df["PRCP"] >= 0.254) & (df["TMEAN"] >= 2)
    )
    df = df.loc[mask]

    if d == "24h":
        df["PRCP"] = df["PRCP"] / 24
        df["VIMC"] = df["VIMC"] / 24
    return df


def pool_pt(stations, d, var, oe_dir):
    """Pool P-T data for a given station list"""
    prcp = []
    exg = []
    fail = 0
    for stn_id in stations:
        try:
            df = load_data(stn_id, d, data_dir=oe_dir)
        except:
            fail += 1
            continue

        prcp.append(df["PRCP"].values)
        exg.append(df[var].values)

    prcp = np.concatenate(prcp)
    exg = np.concatenate(exg)
    return prcp, exg


def c_T(T, c0, beta_c):
    return c0 * np.exp(beta_c * T)


def w_T(T, w0, beta_w):
    return w0 * np.exp(beta_w * T)


def neg_log_likelihood_wT(params, x, T):
    c0, beta_c, w0, beta_w = params
    ct = c_T(T, c0, beta_c)
    wt = w_T(T, w0, beta_w)
    log_likelihood = (
        np.log(wt) - np.log(ct) + (wt - 1) * np.log(x / ct) - (x / ct) ** wt
    )
    return -np.sum(log_likelihood)


def neg_log_likelihood_w_const(params, x, T):
    A, b, w = params
    ct = c_T(T, A, b)
    log_likelihood = np.log(w) - np.log(ct) + (w - 1) * np.log(x / ct) - (x / ct) ** w
    return -np.sum(log_likelihood)


def T_histogram(T_bins, temp_data, pmf=True, ax=None):
    T_midbin = (T_bins[1:] + T_bins[:-1]) / 2
    T_freq, bin_edges = np.histogram(temp_data, bins=T_bins, density=False)
    if pmf:
        T_freq = T_freq / np.sum(T_freq)  # pdf -> pmf
    if ax:
        ax.stairs(T_freq, edges=bin_edges)
        ax.set_xlabel("T [°C]")
        ax.set_ylabel("Frequency")
        ax.set_title("Event Temperature Distribution")
    return T_freq, T_midbin


def wei_cond_cdf_varying(x, T, cw_reg, c=None, w=None):
    """Allow c or w to be fixed"""
    ct = (
        c * np.ones(len(T)) if c is not None else c_T(T, cw_reg["c0"], cw_reg["beta_c"])
    )
    wt = (
        w * np.ones(len(T)) if w is not None else w_T(T, cw_reg["w0"], cw_reg["beta_w"])
    )
    x = np.maximum(x, 0)  # To avoid error during optimization (in mev_PT_cdf)
    return 1 - np.exp(-((x / ct) ** wt))


def exponential_reg_ML(sample, temp, initial_guess=None, wt=True, return_aic=False):
    """wt=True refers to the case where w is assume to be a function of temperature"""
    x = sample[sample > 0]
    T = temp[sample > 0]
    if wt:
        initial_guess = (
            [2.5, 0.07, 1.5, -0.02] if initial_guess is None else initial_guess
        )
        # initial_guess = [1, 0.01, 1, -0.001]
        bnds = ((0, None), (None, None), (0, None), (None, None))
        result = minimize(
            neg_log_likelihood_wT,
            initial_guess,
            args=(x, T),
            method="Nelder-Mead",
            bounds=bnds,
        )
        c0, beta_c, w0, beta_w = result.x
        cw_reg = {
            "c0": c0,
            "beta_c": beta_c,
            "w0": w0,
            "beta_w": beta_w,
        }
        if return_aic:
            k = len(initial_guess)
            nll = neg_log_likelihood_wT(result.x, x, T)
    else:
        initial_guess = [2.5, 0.07, 1.5] if initial_guess is None else initial_guess
        bnds = ((0, None), (None, None), (0, None))
        result = minimize(
            neg_log_likelihood_w_const,
            initial_guess,
            args=(x, T),
            method="L-BFGS-B",
            bounds=bnds,
        )
        c0, beta_c, w = result.x
        cw_reg = {
            "c0": c0,
            "beta_c": beta_c,
            "w0": w,
            "beta_w": 0,
        }
        if return_aic:
            k = len(initial_guess)
            nll = neg_log_likelihood_w_const(result.x, x, T)
    if return_aic:
        aic = 2 * k + 2 * nll
        return cw_reg, aic
    else:
        return cw_reg


def mev_PT_cdf(x, T_midbin, T_freq, n_mean, wei_cond_cdf):
    """
    Return the non exceedance probability by considering all possible temperature bins
    F(x) = sum{F(x|Ti)*g(Ti)}
    T_midbin = Mid point of corresponding T_freq
    T_freq = Freqency
    """
    ### Compute F(x|Ti) for every Ti
    cond_cdf = wei_cond_cdf(x, T_midbin)

    ### Extremes
    return np.sum(cond_cdf * T_freq) ** n_mean


def mev_PT_icdf_robust(
    F, T_midbin, T_freq, n_mean, wei_cond_cdf, x0=None, method="auto"
):
    """
    Robust inverse CDF calculation with multiple fallback strategies

    Parameters:
    -----------
    F : float
        Target probability (0 < F < 1)
    T_midbin, T_freq, n_mean, wei_cond_cdf : as before
    x0 : float or None
        Initial guess. If None, will be estimated automatically
    method : str
        'auto', 'fsolve', 'brentq', or 'minimize'

    Returns:
    --------
    float : The inverse CDF value
    """

    # Input validation
    if not (0 < F < 1):
        raise ValueError("F must be between 0 and 1")

    # Define the objective function
    def objective(x):
        return mev_PT_cdf(x, T_midbin, T_freq, n_mean, wei_cond_cdf) - F

    # Method 1: Improved initial guess
    if x0 is None:
        x0 = estimate_initial_guess(F, T_midbin, T_freq, n_mean)

    # Method selection
    if method == "auto":
        # Try methods in order of preference
        methods_to_try = ["fsolve_robust", "brentq", "minimize"]
    else:
        methods_to_try = [method]

    for method_name in methods_to_try:
        try:
            if method_name == "fsolve_robust":
                return fsolve_robust_approach(objective, x0)
            elif method_name == "brentq":
                return brentq_approach(objective, x0)
            elif method_name == "minimize":
                return minimize_approach(objective, x0)
            elif method_name == "fsolve":
                return fsolve_basic_approach(objective, x0)
        except Exception as e:
            print(f"Method {method_name} failed: {e}")
            continue

    raise RuntimeError("All methods failed to converge")


def estimate_initial_guess(F, T_midbin, T_freq, n_mean):
    """
    Estimate a better initial guess based on the problem structure
    """
    # Use a simple approximation: assume uniform temperature distribution
    # and use inverse Weibull as starting point
    T_mean = np.average(T_midbin, weights=T_freq)

    # Rough approximation based on typical Weibull parameters
    # This can be refined based on your specific application
    if F < 0.1:
        x0 = 0.1
    elif F > 0.99:
        x0 = 100.0
    else:
        # Use quantile transformation as rough estimate
        x0 = -np.log(1 - F ** (1 / n_mean)) * (10 + T_mean / 10)

    return max(x0, 1e-6)  # Ensure positive


def fsolve_robust_approach(objective, x0):
    """
    Enhanced fsolve with better parameters and error handling
    """
    # Try multiple initial guesses if first fails
    initial_guesses = [x0, x0 * 0.1, x0 * 10, 1.0, 10.0]

    for guess in initial_guesses:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = fsolve(
                    objective,
                    x0=guess,
                    xtol=1e-12,  # Tighter tolerance
                    maxfev=1000,  # More function evaluations
                    factor=0.1,  # Smaller initial step
                )

            # Verify the solution
            if abs(objective(result[0])) < 1e-6:
                return result[0]

        except Exception:
            continue

    raise RuntimeError("fsolve failed with all initial guesses")


def brentq_approach(objective, x0):
    """
    Use Brent's method (requires bracketing the root)
    """
    # Find brackets for the root
    a, b = find_brackets(objective, x0)

    # Use Brent's method
    result = brentq(objective, a, b, xtol=1e-12)
    return result


def find_brackets(objective, x0):
    """
    Find brackets [a, b] such that objective(a) and objective(b) have opposite signs
    """
    # Start from x0 and expand outward
    a = max(x0 * 0.001, 1e-6)  # Lower bound (must be positive)
    b = x0 * 1000  # Upper bound

    # Adjust bounds to ensure opposite signs
    fa = objective(a)
    fb = objective(b)

    # Expand bounds if needed
    max_iterations = 50
    for i in range(max_iterations):
        if fa * fb < 0:  # Opposite signs found
            break

        if abs(fa) < abs(fb):
            # Expand lower bound
            a = a * 0.1
            fa = objective(a)
        else:
            # Expand upper bound
            b = b * 10
            fb = objective(b)

    if fa * fb >= 0:
        raise ValueError("Could not find suitable brackets for root finding")

    return a, b


def minimize_approach(objective, x0):
    """
    Use minimization of |objective(x)|^2 as fallback
    """

    def objective_squared(x):
        return objective(x) ** 2

    result = minimize_scalar(objective_squared, bounds=(1e-6, 1000), method="bounded")

    if result.success and abs(objective(result.x)) < 1e-6:
        return result.x
    else:
        raise RuntimeError("Minimization approach failed")


def fsolve_basic_approach(objective, x0):
    """
    Basic fsolve approach (original method)
    """
    result = fsolve(objective, x0=x0)
    return result[0]


def mev_PT_bootstrap_ci(df, var, Fi_val, ntimes=1000, confidence_level=0.95):
    """
    Estimate confidence intervals for MEV quantile estimates using bootstrap resampling.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    var : str
        Variable name for exogenous variable
    Fi_val : array-like
        Return periods or frequencies for quantile estimation
    ntimes : int, default=1000
        Number of bootstrap iterations
    confidence_level : float, default=0.95
        Confidence level for intervals
    mev_boot_yearly : callable, optional
        Custom bootstrap function
    T_bins : array-like, optional
        Temperature bins
    d : str, optional
        Duration parameter

    Returns:
    --------
    dict
        Dictionary containing mean quantiles, confidence intervals, and success count
    """

    Fi_val = np.asarray(Fi_val)
    m = len(Fi_val)
    QM = np.zeros((ntimes, m))

    # Calculate confidence bounds
    alpha = 1 - confidence_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)

    print(f"Starting bootstrap with {ntimes} iterations...")
    successful_iterations = 0

    with tqdm(total=ntimes, desc="Bootstrap Progress", unit="iter") as pbar:
        for ii in range(ntimes):
            try:
                dfr = mev_boot_yearly(df)

                # Fit MEV and estimate quantiles
                quantiles_boot = fit_mev_and_estimate_quantiles(dfr, var, Fi_val)
                QM[ii, :] = quantiles_boot
                successful_iterations += 1

                success_rate = successful_iterations / (ii + 1) * 100
                pbar.set_postfix(
                    {
                        "Success": f"{successful_iterations}/{ii+1}",
                        "Rate": f"{success_rate:.1f}%",
                    }
                )

            except Exception as e:
                QM[ii, :] = np.nan
                # print(f"Iteration {ii+1} failed: {str(e)}")

            pbar.update(1)

    # Remove failed iterations
    valid_mask = ~np.isnan(QM).any(axis=1)
    QM_valid = QM[valid_mask, :]

    print(
        f"\nBootstrap completed: {successful_iterations}/{ntimes} successful iterations"
    )

    if successful_iterations == 0:
        raise RuntimeError(
            "All bootstrap iterations failed. Check your data and model setup."
        )

    if successful_iterations < ntimes * 0.1:  # Less than 10% success rate
        print(
            "Warning: Low success rate. Consider checking your data quality and model parameters."
        )

    # Calculate confidence intervals
    Q_low = np.zeros(m)
    Q_up = np.zeros(m)
    Q_mean = np.zeros(m)

    for jj in range(m):
        qi = QM_valid[:, jj]
        Q_low[jj] = np.percentile(qi, lower_percentile)
        Q_up[jj] = np.percentile(qi, upper_percentile)
        Q_mean[jj] = np.mean(qi)

    return {
        "Q_mean": Q_mean,
        "Q_lower": Q_low,
        "Q_upper": Q_up,
        "n_successful": successful_iterations,
    }


def mev_boot_yearly(df):
    """
    Non-parametric bootstrap technique for MEV that:
    1) Resamples the number of events for each year in the series
    2) Resamples the daily events within each sampled year
    Both steps use sampling with replacement.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with datetime index containing precipitation and other variables

    Returns:
    --------
    pandas.DataFrame
        Bootstrap sample with proper datetime index maintaining chronological order
    """
    years = df.index.year.unique()
    dfr_list = []

    for target_year in years:
        # Sample a source year at random (with replacement)
        source_year = np.random.choice(years)
        source_data = df.loc[df.index.year == source_year].copy()

        if len(source_data) == 0:
            continue

        # Bootstrap sample the daily events within that year
        n_events = len(source_data)
        boot_indices = np.random.choice(range(n_events), size=n_events, replace=True)
        boot_data = source_data.iloc[boot_indices].copy()

        # # Create new datetime index for target year (by simply preserve the day-of-year structure but update the year)
        # new_index = source_data.index.to_series().apply(lambda d: d.replace(year=target_year)).values
        # boot_data.index = new_index

        dfr_list.append(boot_data)

    result = pd.concat(dfr_list, ignore_index=False)
    return result


def fit_mev_and_estimate_quantiles(df, var, F):
    prcp = df["PRCP"].values
    exg = df[var].values

    cw_scaling = scaling_analysis(prcp, exg, var, ci=True, ntimes=1000, n_bins=20)

    threshold = np.percentile(prcp, 10)
    c_reg = linregress(cw_scaling[var + "_med"], cw_scaling["c"])
    w_reg = linregress(cw_scaling[var + "_med"], np.log10(cw_scaling["w"]))
    initial_guess = [
        np.exp(c_reg.intercept),
        c_reg.slope,
        np.exp(w_reg.intercept),
        w_reg.slope,
    ]

    cw_reg = exponential_reg_ML(prcp - threshold, exg, initial_guess=initial_guess)
    # T_bins = np.arange(-0.25, 40, 0.5)
    T_bins = np.arange(-40, 40, 0.5)
    T_freq, T_midbin = T_histogram(T_bins, exg, pmf=True, ax=None)

    N_event_p_y = df["PRCP"].resample("YE").count().mean()

    cond_cdf = partial(wei_cond_cdf_varying, cw_reg=cw_reg, c=None, w=None)

    quant_MEV = np.array(
        [
            mev_PT_icdf_robust(Fi, T_midbin, T_freq, N_event_p_y, cond_cdf) + threshold
            for Fi in F
        ]
    )
    return quant_MEV.flatten()


def get_annual_max(df_, eps_scale=1e-3):
    df = df_.copy()

    # Tie Breaker
    eps = eps_scale * np.random.uniform(0, 1, len(df))
    df["PRCP"] = df["PRCP"] + eps

    am = df["PRCP"].resample("YE").max().values
    am = np.sort(am[~np.isnan(am)])
    F = (np.arange(len(am)) + 1) / (len(am) + 1)

    return am, F
