# groundwater_hds.py
import os
import numpy as np
import xarray as xr
import pandas as pd

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[GroundwaterHDS] scipy,‘’,.")


def _fill_nans_by_nearest_sample(values, lons, lats, label_prefix="[GroundwaterHDS]"):
    """Within the "collapse point sample set", NaNs are padded with the nearest valuable sample.

    values: one-dimensional array, length = n_points (HDS of a certain year)
    lons, lats: the latitude and longitude of the corresponding point
    Return: filled values (new array, not modified in place)"""
    values = np.asarray(values, dtype="float64")
    lons = np.asarray(lons, dtype="float64")
    lats = np.asarray(lats, dtype="float64")

    if not HAS_SCIPY:
        print(f"{label_prefix}scipy is not installed, skip the nearest sample point complement and keep NaN.")
        return values

    mask_valid = np.isfinite(values)
    if mask_valid.all():
        # ,
        return values
    if not np.any(mask_valid):
        # They are all NaN and cannot be corrected.
        print(f"{label_prefix}Warning: All points in this year are NaN, and the nearest point complement cannot be done.")
        return values

    coords_valid = np.column_stack([lons[mask_valid], lats[mask_valid]])
    coords_nan = np.column_stack([lons[~mask_valid], lats[~mask_valid]])

    tree = cKDTree(coords_valid)
    dist, idx = tree.query(coords_nan, k=1)

    filled = values.copy()
    filled[~mask_valid] = values[mask_valid][idx]

    return filled


def _extract_hds_from_zarr(store_path, years, sinkhole_position, label_prefix="[GroundwaterHDS]"):
    """From the given Zarr dataset, extract the values of each collapse point for a specified list of years.
    Annual average HDS (Groundwater Head), returns {year: np.ndarray}.

    Steps:
    1. For a specified year, extract the corresponding year from the time dimension and find the annual average -> 2D field (lat, lon)
    2. Use xarray’s .sel(lat, lon, method="nearest") to interpolate each collapse point
    3. For points that are still NaN after interpolation, fill them with "the most recent sample point with value" (Requirement 10.1)"""

    if not years:
        return {}

    if not os.path.exists(store_path):
        raise FileNotFoundError(f"{label_prefix}Zarr dataset not found:{store_path}")

    print(f"{label_prefix}Open the Zarr dataset:{store_path}")

    ds = xr.open_zarr(store_path)
    try:
        # 1. Select HDS variable: priority l1_hds -> hds -> HDS -> l2_hds
        candidate_vars = ["l1_hds", "hds", "HDS", "l2_hds"]
        hds_var = None
        for v in candidate_vars:
            if v in ds.data_vars:
                hds_var = v
                break
        if hds_var is None:
            hds_var = list(ds.data_vars)[0]
            print(f"{label_prefix}Warning: l1_hds/l2_hds not found, using variable{hds_var}")

        da = ds[hds_var]

        # 2. Unify longitude and latitude names: change latitude/longitude to lat/lon (either coordinates or dimensions)
        rename_dict = {}
        if "latitude" in ds.coords or "latitude" in ds.dims:
            rename_dict["latitude"] = "lat"
        if "longitude" in ds.coords or "longitude" in ds.dims:
            rename_dict["longitude"] = "lon"
        if rename_dict:
            ds = ds.rename(rename_dict)
            da = ds[hds_var]

        if "lat" not in ds.coords or "lon" not in ds.coords:
            raise ValueError(f"{label_prefix}'lat'/'lon' , Zarr .")

        # 3. Time -> Year
        if "time" not in ds.coords:
            raise ValueError(f"{label_prefix}The 'time' coordinate is missing from the dataset.")

        time_coord = ds["time"]
        tvals = time_coord.values
        if np.issubdtype(tvals.dtype, np.datetime64):
            years_all = pd.to_datetime(tvals).year
        else:
            years_all = np.array(tvals).astype(int)

        # 4. Prepare the coordinates of the collapse point
        lats_pts = sinkhole_position["Latitude"].values
        lons_pts = sinkhole_position["Longitude"].values
        pts_lat = xr.DataArray(lats_pts, dims="points")
        pts_lon = xr.DataArray(lons_pts, dims="points")

        result = {}

        for year in years:
            idxs = np.where(years_all == year)[0]
            if idxs.size == 0:
                raise ValueError(
                    f"{label_prefix}: {store_path} has no time entry for year {year}."
                )

            print(f"{label_prefix}Processing year{year} ...")
            da_year = da.isel(time=idxs)
            da_year_mean = da_year.mean(dim="time")

            # 5. Do standard nearest neighbor interpolation first
            sample = da_year_mean.sel(lat=pts_lat, lon=pts_lon, method="nearest")
            vals = sample.values.astype("float64")
            vals = np.where(np.isfinite(vals), vals, np.nan)

            # 6. Fill NaN with the most recent valuable sample "within the sample set" (Requirement 10.1)
            vals_filled = _fill_nans_by_nearest_sample(
                vals, lons_pts, lats_pts, label_prefix=f"{label_prefix}-year{year}"
            )

            result[year] = vals_filled

        return result

    finally:
        ds.close()


def groundwater_hds(
    sinkhole_position,
    database_folder_path,
    historical_folder_path,
    future_ssp_folder_path,
    ssp,
):
    """Groundwater Head HDS (Groundwater Head) extraction and historical/future time period average calculation (including 10.1 recent sample point supplementary values).

    data structure
    --------
    Root directory:
      Z:\\jing\\Large_scale\\future_dataset\\18_groundwater_1960_2100_1km\\Annual

    · History (1960–2014):
      Annual\\historical\\hds_annual_1960_2014_historical_ensemble.zarr

    · Future SSP (2015–2100):
      Annual\\sspxxx\\hds_annual_2015_2100_sspxxx_ensemble.zarr

      The mapping between sspxxx and ssp (ignoring case) is:
        ssp=ssp1 -> ssp126
        ssp=ssp3 -> ssp370
        ssp=ssp5 -> ssp585

    Calculation rules
    --------
    · Each year: using the HDS annual average for that year (meaning the time dimension),
      Then interpolate the collapse point + complement the nearest sample point.

    · History:
        Average HDS using three years 2000, 2010, 2020:
        - 2000, 2010 from historical
        - 2020 from the corresponding SSP dataset (2015+)
        Output column: HDS_hist_2000_2010_2020

    · Future:
        Average over a 20-year period and 10-year intervals:
        - 2020–2040: 2020, 2030, 2040
        - 2040–2060: 2040, 2050, 2060
        - 2060–2080: 2060, 2070, 2080
        - 2080–2100: 2080, 2090, 2100

        Output columns:
        HDS_2020_2040, HDS_2040_2060, HDS_2060_2080, HDS_2080_2100"""

    print("\\n[GroundwaterHDS] Start calculating groundwater head (HDS)...")

    # ---------- 1. Check the input column ----------
    required_cols = ["No", "Longitude", "Latitude"]
    for col in required_cols:
        if col not in sinkhole_position.columns:
            raise ValueError(
                f"[GroundwaterHDS] The input sinkhole_position is missing a required column: '{col}'"
            )

    # ---------- 2. SSP -> sspxxx ----------
    ssp_to_sspxxx = {
        "ssp1": "ssp126",
        "ssp3": "ssp370",
        "ssp5": "ssp585",
    }
    ssp_key = ssp.lower()
    if ssp_key not in ssp_to_sspxxx:
        raise ValueError(
            f"[GroundwaterHDS] Unsupported SSP scenarios:{ssp},{list(ssp_to_sspxxx.keys())}"
        )
    sspxxx = ssp_to_sspxxx[ssp_key]  # Such as 'ssp3' -> 'ssp370'

    # ---------- 3. Construct path ----------
    gw_root = os.path.join(
        database_folder_path,
        "18_groundwater_1960_2100_1km",
        "Annual",
    )

    # Zarr
    hist_store = os.path.join(
        gw_root,
        "historical",
        "hds_annual_1960_2014_historical_ensemble.zarr",
    )

    # Future SSP Zarr: Annual/sspxxx/hds_annual_2015_2100_sspxxx_ensemble.zarr
    ssp_dir = os.path.join(gw_root, sspxxx)
    ssp_store = os.path.join(
        ssp_dir,
        f"hds_annual_2015_2100_{sspxxx}_ensemble.zarr",
    )

    print(f"[GroundwaterHDS] Zarr:{hist_store}")
    print(f"[GroundwaterHDS] SSP   Zarr: {ssp_store}")
    print(f"[GroundwaterHDS] Current SSP:{ssp} -> {sspxxx}")

    # ---------- 4. Required year ----------
    hist_years = [2000, 2010, 2020]

    future_windows = {
        "HDS_2020_2040": [2020, 2030, 2040],
        "HDS_2040_2060": [2040, 2050, 2060],
        "HDS_2060_2080": [2060, 2070, 2080],
        "HDS_2080_2100": [2080, 2090, 2100],
    }

    all_future_years = sorted({y for ys in future_windows.values() for y in ys})
    years_needed = sorted(set(hist_years + all_future_years))

    hist_years_store = [y for y in years_needed if y < 2015]   # Use history zarr
    ssp_years_store = [y for y in years_needed if y >= 2015]   # SSP zarr

    # ---------- 5. Extract HDS of each year from Zarr ----------
    year_values = {}

    hist_map = _extract_hds_from_zarr(
        hist_store,
        hist_years_store,
        sinkhole_position,
        label_prefix="[GroundwaterHDS-HIST]",
    )
    year_values.update(hist_map)

    ssp_map = _extract_hds_from_zarr(
        ssp_store,
        ssp_years_store,
        sinkhole_position,
        label_prefix="[GroundwaterHDS-SSP]",
    )
    year_values.update(ssp_map)

    missing_years = [y for y in years_needed if y not in year_values]
    if missing_years:
        raise RuntimeError(
            f"[GroundwaterHDS] Zarr :{missing_years}"
        )

    # ---------- 6. Historical average (2000, 2010, 2020) ----------
    print("[GroundwaterHDS] Calculate historical average (2000, 2010, 2020)...")
    hist_stack = np.vstack([year_values[y] for y in hist_years])
    hist_mean = np.nanmean(hist_stack, axis=0)
    hist_col = "HDS_hist_2000_2010_2020"
    sinkhole_position[hist_col] = hist_mean

    # ---------- 7. Average for each time period in the future ----------
    for col_name, ys in future_windows.items():
        print(f"[GroundwaterHDS] Calculate future time period{col_name}Corresponding year{ys}...")
        stack = np.vstack([year_values[y] for y in ys])
        mean_vals = np.nanmean(stack, axis=0)
        sinkhole_position[col_name] = mean_vals

    # ---------- 8. Save the result ----------
    os.makedirs(historical_folder_path, exist_ok=True)
    os.makedirs(future_ssp_folder_path, exist_ok=True)

    # History CSV
    hist_out_cols = ["No", "Longitude", "Latitude", hist_col]
    hist_out_cols = [c for c in hist_out_cols if c in sinkhole_position.columns]
    hist_df = sinkhole_position[hist_out_cols].copy()
    hist_output_path = os.path.join(
        historical_folder_path,
        "GroundwaterHDS_historical_2000_2010_2020.csv",
    )
    hist_df.to_csv(hist_output_path, index=False, encoding="utf-8-sig")

    # Future CSV
    future_out_cols = ["No", "Longitude", "Latitude"] + list(future_windows.keys())
    future_out_cols = [c for c in future_out_cols if c in sinkhole_position.columns]
    future_df = sinkhole_position[future_out_cols].copy()
    future_output_path = os.path.join(
        future_ssp_folder_path,
        f"GroundwaterHDS_future_{ssp}.csv",
    )
    future_df.to_csv(future_output_path, index=False, encoding="utf-8-sig")

    # ---------- 9. Print simple statistics ----------
    print("\\n[GroundwaterHDS] Historical data statistics (average of 2000, 2010, 2020):")
    if not hist_df[hist_col].isna().all():
        print(f"Points:{len(hist_df)}")
        print(f"Minimum value:{hist_df[hist_col].min():.4f}")
        print(f":{hist_df[hist_col].max():.4f}")
        print(f"Average:{hist_df[hist_col].mean():.4f}")

    print("\\n[GroundwaterHDS] Statistics for each time period in the future:")
    for col_name in future_windows.keys():
        col_series = future_df[col_name]
        if col_series.isna().all():
            continue
        print(f"  {col_name}:")
        print(f":{len(col_series)}")
        print(f":{col_series.min():.4f}")
        print(f"Maximum value:{col_series.max():.4f}")
        print(f"Average:{col_series.mean():.4f}")

    print("\\n[GroundwaterHDS] :")
    print("  ", hist_output_path)
    print("[GroundwaterHDS] :")
    print("  ", future_output_path)

    return sinkhole_position
