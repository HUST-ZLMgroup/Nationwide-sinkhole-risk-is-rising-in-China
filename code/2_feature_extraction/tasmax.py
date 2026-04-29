# tasmax.py
import os
import numpy as np
import xarray as xr


def _extract_tasmax_for_year(nc_path, sinkhole_position, label=None):
    """Extracts the average maximum temperature for the year (averaged over the time dimension) for each point from the given NetCDF file.

    The return value is a one-dimensional numpy array with the same length as sinkhole_position.
    If the variable unit is K/Kelvin, convert to ℃."""
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"[Tasmax] NetCDF file not found:{nc_path}")

    print(f"[Tasmax] Open{label or os.path.basename(nc_path)}: {nc_path}")

    ds = xr.open_dataset(nc_path)
    try:
        # 1. : 'tasmax'
        if "tasmax" in ds.data_vars:
            da = ds["tasmax"]
        else:
            # If there is no tasmax, take the first variable (usually this will not happen, just a cover-up)
            first_var = list(ds.data_vars)[0]
            da = ds[first_var]
            print(f"[Tasmax] : 'tasmax',{first_var}")

        # 2. time ()
        time_dim_candidates = [d for d in da.dims if "time" in d.lower()]
        if time_dim_candidates:
            time_dim = time_dim_candidates[0]
            da_mean = da.mean(dim=time_dim)
        else:
            # time ,
            da_mean = da

        # 3. Find the latitude/longitude dimension name and rename it to lat / lon to facilitate interpolation
        lat_name = next((d for d in da_mean.dims if "lat" in d.lower()), None)
        lon_name = next((d for d in da_mean.dims if "lon" in d.lower()), None)
        if lat_name is None or lon_name is None:
            raise ValueError(
                "[Tasmax] NetCDF /,."
            )

        rename_dict = {}
        if lat_name != "lat":
            rename_dict[lat_name] = "lat"
        if lon_name != "lon":
            rename_dict[lon_name] = "lon"
        da_mean = da_mean.rename(rename_dict)

        # 4. Use nearest neighbor interpolation to map the raster to the latitude and longitude of the collapse point
        lats = sinkhole_position["Latitude"].values
        lons = sinkhole_position["Longitude"].values

        pts_lat = xr.DataArray(lats, dims="points")
        pts_lon = xr.DataArray(lons, dims="points")

        # Processing step.
        sample = da_mean.sel(lat=pts_lat, lon=pts_lon, method="nearest")
        vals = sample.values.astype("float64")

        # 5. Unit conversion: if it is K / Kelvin, convert to ℃
        units_raw = str(da.attrs.get("units", "")).strip().lower()
        units_norm = units_raw.replace(" ", "")
        if (
            units_norm in {"k", "degk", "degreekelvin", "degreeskelvin"}
            or "kelvin" in units_norm
        ):
            vals = vals - 273.15

        # 6. Set non-finite value to NaN
        vals = np.where(np.isfinite(vals), vals, np.nan)

        return vals
    finally:
        ds.close()



def tasmax(
    sinkhole_position,
    database_folder_path,
    historical_folder_path,
    future_ssp_folder_path,
    ssp,
):
    """Calculate maximum temperature (annual average) and take multi-year monthly averages over historical/future time periods.

    data path
    --------
    The resolution and time organization are consistent with precipitation data, and the files are divided by year.

    - Examples of historical data before 2015 (excluding 2015):
      Z:\\jing\\Large_scale\\future_dataset\\21_tasmax\\historical\\
      tasmax_day_BCC-CSM2-MR_historical_r1i1p1f1_gn_2000_v2.0.nc

      Assuming other year filename patterns:
      tasmax_day_BCC-CSM2-MR_historical_r1i1p1f1_gn_{year}_v2.0.nc

    - Examples of SSP data for 2015 and later (inclusive):
      Z:\\jing\\Large_scale\\future_dataset\\21_tasmax\\sspxxx\\
      tasmax_day_BCC-CSM2-MR_sspxxx_r1i1p1f1_gn_2017_v2.0.nc

      Assuming other year filename patterns:
      tasmax_day_BCC-CSM2-MR_{sspxxx}_r1i1p1f1_gn_{year}_v2.0.nc

      The mapping relationship between sspxxx and ssp variables (ignoring case) is:
        ssp=ssp1 -> ssp126
        ssp=ssp2 -> ssp245
        ssp=ssp3 -> ssp370
        ssp=ssp5 -> ssp585

    Calculation rules
    --------
    1) For each year file, first average the time dimension to obtain the annual average maximum temperature field of that year.

    2) Historical data:
       Use the annual average maximum temperature in 2000, 2010, and 2020, and then average:
       Tasmax_hist_2000_2010_2020

       Among them:
         - 2000, 2010 from historical directory
         - 2020 from the corresponding SSP directory (take SSP in 2015 and later)

    3) Future data:
       Taking 20 years as a time period, using the annual average maximum temperature in 10-year intervals, the three-year average:
       - 2020-2040: Use 2020, 2030, 2040
       - 2040-2060: Use 2040, 2050, 2060
       - 2060-2080: Use 2060, 2070, 2080
       - 2080-2100: Use 2080, 2090, 2100

       Corresponding output column:
       Tasmax_2020_2040, Tasmax_2040_2060, Tasmax_2060_2080, Tasmax_2080_2100

    parameters
    ----
    sinkhole_position : pandas.DataFrame
        Contains at least the columns 'No', 'Longitude', 'Latitude'
    database_folder_path : str
        Database root directory, for example Z:\\jing\\Large_scale\\future_dataset
    historical_folder_path : str
        Historical data output directory (.../historical)
    future_ssp_folder_path : str
        Future data output directory (.../future/sspX)
    ssp : str
        SSP context string, such as 'ssp1', 'ssp2', 'ssp3', 'ssp5'

    Return
    ----
    pandas.DataFrame
        Add the following columns to the original DataFrame:
        - Tasmax_hist_2000_2010_2020
        - Tasmax_2020_2040
        -Tasmax_2040_2060
        - Tasmax_2060_2080
        - Tasmax_2080_2100

    Description
    ----
    If the original tasmax unit is K/Kelvin, the result is automatically converted to ℃;
    If the original unit is already ℃ or other units, keep the original value."""

    print("\\n[Tasmax] Start calculating the maximum temperature...")

    # ---------------- 1. Check the input column ----------------
    required_cols = ["No", "Longitude", "Latitude"]
    for col in required_cols:
        if col not in sinkhole_position.columns:
            raise ValueError(f"[Tasmax] The input sinkhole_position is missing a required column: '{col}'")

    # ---------------- 2. SSP -> sspxxx mapping ----------------
    ssp_to_sspxxx = {
        "ssp1": "ssp126",
        "ssp2": "ssp245",
        "ssp3": "ssp370",
        "ssp5": "ssp585",
    }

    ssp_key = ssp.lower()
    if ssp_key not in ssp_to_sspxxx:
        raise ValueError(
            f"[Tasmax] Unsupported SSP scenarios:{ssp}, currently only supports{list(ssp_to_sspxxx.keys())}"
        )

    sspxxx = ssp_to_sspxxx[ssp_key]  # Such as 'ssp3' -> 'ssp370'

    # ---------------- 3. Construct the data path root directory ----------------
    tasmax_root = os.path.join(
        database_folder_path,
        "21_tasmax",
    )

    # Directory where historical nc is located
    hist_dir = os.path.join(tasmax_root, "historical")

    # SSP nc directory:.../sspxxx/
    ssp_dir = os.path.join(tasmax_root, sspxxx)

    print(f"[Tasmax] Historical data directory:{hist_dir}")
    print(f"[Tasmax] SSP projection data directory:{ssp_dir}")
    print(f"[Tasmax] Current SSP scenario:{ssp} -> {sspxxx}")

    # ---------------- 4. ----------------
    hist_years = [2000, 2010, 2020]

    future_windows = {
        "Tasmax_2020_2040": [2020, 2030, 2040],
        "Tasmax_2040_2060": [2040, 2050, 2060],
        "Tasmax_2060_2080": [2060, 2070, 2080],
        "Tasmax_2080_2100": [2080, 2090, 2100],
    }

    all_future_years = sorted({y for years in future_windows.values() for y in years})
    years_needed = sorted(set(hist_years + all_future_years))

    # ---------------- 5. Extract the annual average maximum temperature for each year ----------------
    year_values = {}

    for year in years_needed:
        if year < 2015:
            # tasmax_day_BCC-CSM2-MR_historical_r1i1p1f1_gn_{year}_v2.0.nc
            nc_name = f"tasmax_day_BCC-CSM2-MR_historical_r1i1p1f1_gn_{year}_v2.0.nc"
            nc_path = os.path.join(hist_dir, nc_name)
        else:
            # tasmax_day_BCC-CSM2-MR_{sspxxx}_r1i1p1f1_gn_{year}_v2.0.nc
            nc_name = f"tasmax_day_BCC-CSM2-MR_{sspxxx}_r1i1p1f1_gn_{year}_v2.0.nc"
            nc_path = os.path.join(ssp_dir, nc_name)

        label = f"{year}"
        year_values[year] = _extract_tasmax_for_year(
            nc_path, sinkhole_position, label=label
        )

    # ---------------- 6. History: 2000, 2010, 2020 average again ----------------
    print("[Tasmax] Calculate historical average (2000, 2010, 2020)...")
    hist_stack = np.vstack([year_values[y] for y in hist_years])
    hist_mean = np.nanmean(hist_stack, axis=0)
    hist_col = "Tasmax_hist_2000_2010_2020"
    sinkhole_position[hist_col] = hist_mean

    # ---------------- 7. Average in each future time period ----------------
    for col_name, years in future_windows.items():
        print(f"[Tasmax] Calculate future time period{col_name}Corresponding year{years}’s average annual maximum temperature average...")
        stack = np.vstack([year_values[y] for y in years])
        mean_vals = np.nanmean(stack, axis=0)
        sinkhole_position[col_name] = mean_vals

    # ---------------- 8. Save historical and future results ----------------
    hist_out_cols = ["No", "Longitude", "Latitude", hist_col]
    hist_out_cols = [c for c in hist_out_cols if c in sinkhole_position.columns]
    hist_df = sinkhole_position[hist_out_cols].copy()

    os.makedirs(historical_folder_path, exist_ok=True)
    hist_output_path = os.path.join(
        historical_folder_path, "Tasmax_historical_2000_2010_2020.csv"
    )
    hist_df.to_csv(hist_output_path, index=False, encoding="utf-8-sig")

    future_out_cols = ["No", "Longitude", "Latitude"] + list(future_windows.keys())
    future_out_cols = [c for c in future_out_cols if c in sinkhole_position.columns]
    future_df = sinkhole_position[future_out_cols].copy()

    os.makedirs(future_ssp_folder_path, exist_ok=True)
    future_output_path = os.path.join(
        future_ssp_folder_path, f"Tasmax_future_{ssp}.csv"
    )
    future_df.to_csv(future_output_path, index=False, encoding="utf-8-sig")

    # ---------------- 9. Print statistical information ----------------
    print("\\n[Tasmax] Historical data statistics (average annual maximum temperature in 2000, 2010, 2020):")
    if not hist_df[hist_col].isna().all():
        print(f"Points:{len(hist_df)}")
        print(f"Minimum value:{hist_df[hist_col].min():.4f}")
        print(f":{hist_df[hist_col].max():.4f}")
        print(f"Average:{hist_df[hist_col].mean():.4f}")

    print("\\n[Tasmax] ( SSP ):")
    for col_name in future_windows.keys():
        col_series = future_df[col_name]
        if col_series.isna().all():
            continue
        print(f"  {col_name}:")
        print(f":{len(col_series)}")
        print(f":{col_series.min():.4f}")
        print(f"Maximum value:{col_series.max():.4f}")
        print(f"Average:{col_series.mean():.4f}")

    print("\\n[Tasmax] Historical results have been saved to:")
    print("  ", hist_output_path)
    print("[Tasmax] :")
    print("  ", future_output_path)
    print("\\n[Tasmax] ():")
    print(hist_df.head())
    print("\\n[Tasmax] Result preview (future part):")
    print(future_df.head())

    return sinkhole_position
