import xarray as xr
import pandas as pd
import fsspec
import cdsapi
import glob
import os
import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


EXTENT = [49.5, -125, 24.5, -66.75]  # [north, west, south, east]
DATASET = "reanalysis-era5-pressure-levels"
VAR_LIST = ["vertical_velocity"]
PREFIX = "ERA5_CONUS_W500"
# DATASET = "reanalysis-era5-single-levels"
# VAR_LIST = ["vertically_integrated_moisture_divergence"]
# PREFIX = "ERA5_CONUS_VIMD"
FILEPATH = "/mnt/d/climate_data/ERA5_CONUS/"
MAX_SIMULTANEOUS_REQUESTS = 1


def get_month(month, client):
    year, mon = month.split("-")
    days_in_month = calendar.monthrange(int(year), int(mon))[1]
    output_path = os.path.join(FILEPATH, f"{PREFIX}_{month}.grib")

    if os.path.exists(output_path):
        return  # Skip if already exists

    request = {
        "product_type": "reanalysis",
        "pressure_level": ["500"],
        "variable": VAR_LIST,
        "area": EXTENT,  # N, W, S, E
        "year": [year],
        "month": [mon],
        "day": [f"{day:02d}" for day in range(1, days_in_month + 1)],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "data_format": "grib",
    }

    try:
        client.retrieve(DATASET, request, output_path)
    except Exception as e:
        print(f"Failed to download {month}: {e}")


def download_months(months, client):
    with ThreadPoolExecutor(max_workers=MAX_SIMULTANEOUS_REQUESTS) as executor:
        futures = {executor.submit(get_month, month, client): month for month in months}
        for future in as_completed(futures):
            month = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {month}: {e}")


def create_file_list(start_year=1940, end_year=2024):
    months = pd.date_range(
        f"{start_year}-01-01", f"{end_year}-12-31", freq="MS"
    ).strftime("%Y-%m")
    processed_files = glob.glob(os.path.join(FILEPATH, f"{PREFIX}_*.grib"))
    months_processed = {
        os.path.basename(f).replace(f"{PREFIX}_", "").replace(".grib", "")
        for f in processed_files
    }
    return sorted(set(months) - months_processed)


if __name__ == "__main__":
    os.makedirs(FILEPATH, exist_ok=True)
    c = cdsapi.Client()
    months_to_download = create_file_list()
    print(f"{len(months_to_download)} months to download.")

    for i in tqdm(
        range(0, len(months_to_download), MAX_SIMULTANEOUS_REQUESTS), ncols=100
    ):
        batch = months_to_download[i : i + MAX_SIMULTANEOUS_REQUESTS]
        download_months(batch, c)
