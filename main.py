# %%
import pandas as pd
import numpy as np
from pathlib import Path
from src.wei_fun import wei_fit
from src.mplstyle import set as set_mplstyle
from tqdm import tqdm

set_mplstyle("Fira Sans")
import json

DATA_DOC_PATH = Path("./data/dataset_docs/")
MERGED_OE_PATH = Path("./data/OE_tmean_era5/")

OUTPUT_PATH = Path("./output/")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

stations_df = pd.read_csv(DATA_DOC_PATH / "pt_stations_ge30y_80pct_good.csv")
STATIONS = stations_df["StnID"].values


# %%
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


def pwm_estimation(X, threshold=1):
    """
    Perform Parameter Estimation using PWM (Probability-Weighted Moments) method.

    Args:
        X (NumPy 2D array): Events matrix where each row corresponds to a bin
        thresh: Independent events > threshold are used to fit data

    Returns:
        DataFrame: with PWM parameters 'c', 'w', 'T_med'
    """
    X = X - threshold
    pwm_params = np.apply_along_axis(
        wei_fit, axis=1, arr=X, how="pwm", threshold=0
    )  # Negative values are filtered out in wei_fit
    return pwm_params


def scaling_analysis(stn_id, var, durations, threshold=1):
    if not isinstance(durations, list):
        durations = [durations]

    duration_dirs = {d: (OUTPUT_PATH / d) for d in durations}
    for path in duration_dirs.values():
        (path / var).mkdir(parents=True, exist_ok=True)

    for d in durations:
        df = pd.read_csv(
            MERGED_OE_PATH / d / (stn_id + ".csv"), index_col=0, parse_dates=True
        )

        threshold = 1
        mask = (df["PRCP"] >= 0.254) & (df["TMEAN"] >= 2)

        prcp = df[mask]["PRCP"].values
        exg = df[mask][var].values

        # Bin data
        Nbins = 20
        prcp_mat, exg_mat = bin_ps_data(prcp, exg, Nbins)

        # Fit Weibull Distribution at at each bin
        pwm_params = pwm_estimation(prcp_mat, threshold=threshold)
        exg_med = np.median(exg_mat, axis=1)

        # Normalizing parameters
        _, c_o, w_o = wei_fit(prcp - threshold, how="pwm", threshold=0)
        exg_mean = exg.mean()

        # Save results
        cw_scaling = pd.DataFrame(
            {
                "c": pwm_params[:, 1],
                "w": pwm_params[:, 2],
                f"{var}_med": exg_med,
            }
        )
        cw_scaling.to_csv(
            duration_dirs[d] / var / f"cw_scaling_{stn_id}.csv", index=False
        )

        normalization = dict(c0=c_o, w0=w_o, Tmean=exg_mean)
        with open(duration_dirs[d] / var / f"cw_normalization_{stn_id}.json", "w") as f:
            json.dump(normalization, f)


def load_scaling_results(stn_id, var, duration):
    """
    Load the scaling analysis results with TMEAN as exogenous variable for a single duration
    """
    cw_scaling = pd.read_csv(OUTPUT_PATH / duration / var / f"cw_scaling_{stn_id}.csv")
    with open(
        OUTPUT_PATH / duration / var / f"cw_normalization_{stn_id}.json", "r"
    ) as f:
        normalization = json.load(f)
    return cw_scaling, normalization


def main():
    fail = []
    total_iterations = len(STATIONS) * 3  # 3 variables per station
    with tqdm(total=total_iterations) as pbar:
        for stn_id in STATIONS:
            for var in ["TMEAN", "W500", "VIMC"]:
                try:
                    scaling_analysis(
                        stn_id, var=var, durations=["1h", "24h"], threshold=1
                    )
                except Exception as e:
                    fail.append((stn_id, var, e))
                finally:
                    pbar.update(1)

    print(f"{len(STATIONS) - len(fail) // 3}/{len(STATIONS)} analyzed")
    if len(fail) > 0:
        for stn_id, var, e in fail:
            print(
                f"Failed to perform P-{var} scaling analysis at {stn_id} with the following Exception: {e.args[1]}"
            )


if __name__ == "__main__":
    main()
