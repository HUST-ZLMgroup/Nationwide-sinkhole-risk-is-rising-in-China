# leaf_area_index.py
import os
import numpy as np
from rasterio_compat import rasterio
from tqdm import tqdm


def _extract_lai_values_for_month(tif_path, sinkhole_position, label=None):
    """Extracts leaf area index cell values point by point from the given GeoTIFF, returning a numpy array with the same length as sinkhole_position."""
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"[LeafAreaIndex] Raster file not found:{tif_path}")

    desc = f"Extract {label}" if label is not None else f"Extract {os.path.basename(tif_path)}"
    values = []

    with rasterio.open(tif_path) as src:
        nodata = src.nodata

        for _, row in tqdm(
            sinkhole_position.iterrows(),
            total=len(sinkhole_position),
            desc=desc,
        ):
            lon = row["Longitude"]
            lat = row["Latitude"]

            try:
                # For EPSG:4326, index(lon, lat)
                row_idx, col_idx = src.index(lon, lat)
            except Exception:
                values.append(np.nan)
                continue

            # Determine whether it is within the grid range
            if 0 <= row_idx < src.height and 0 <= col_idx < src.width:
                window = rasterio.windows.Window(col_idx, row_idx, 1, 1)
                data = src.read(1, window=window)
                v = float(data[0][0])

                # NoData
                if nodata is not None and (v == nodata or np.isclose(v, nodata)):
                    v = np.nan

                values.append(v)
            else:
                values.append(np.nan)

    return np.array(values, dtype="float64")


def leaf_area_index(
    sinkhole_position,
    database_folder_path,
    historical_folder_path,
    future_ssp_folder_path,
    ssp,
):
    """Extract the Leaf Area Index (LAI) and calculate the average by year and time period.

    Data resolution: 0.05° (~2.5 arc-min), monthly data (12 issues per year).

    data path
    --------
    - Historical data before 2015 (excluding 2015):
      Z:\\jing\\Large_scale\\future_dataset\\1 0_leaf_area_1983_2100_0.05d\\historical\\lai_historical_2xxx_y.tif
      Where 2xxx is the year and y is the month (1-12)

    - Future scenario data in 2015 and beyond (including 2015):
      Z:\\jing\\Large_scale\\future_datasetZXCVBNM9ZXCVB NM10_leaf_area_1983_2100_0.05d\\sspxxx\\lai_sspxxx_2xxx_y.tif

      The mapping relationship between sspxxx and ssp variables (ignoring case) is:
        ssp=ssp1 -> ssp126
        ssp=ssp2 -> ssp245
        ssp=ssp3 -> ssp370
        ssp=ssp5 -> ssp585

    Calculation rules
    --------
    1) First use the average of 3 representative months in each year: April, August, and December
       Annual average LAI = mean(April, August, December)

    2) Historical data:
       Use the annual average LAI of the three years 2000, 2010, and 2020, and then average:
       LAI_hist_2000_2010_2020

       Among them, 2000, 2010 come from the historical path;
            2020 comes from the corresponding SSP path (SSP data in 2015 and later)

    3) Future data:
       Using 20 years as a time period and 10-year intervals, average:
       - 2020-2040: Use the annual average of 2020, 2030, 2040
       - 2040-2060: Use the annual average of 2040, 2050, 2060
       - 2060-2080: Use the annual average of 2060, 2070, 2080
       - 2080-2100: Use the annual average of 2080, 2090, 2100

       Corresponding output column:
       LAI_2020_2040, LAI_2040_2060, LAI_2060_2080, LAI_2080_2100

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
        - LAI_hist_2000_2010_2020
        - LAI_2020_2040
        - LAI_2040_2060
        - LAI_2060_2080
        - LAI_2080_2100"""

    print("\\n[LeafAreaIndex] (LAI) ...")

    # ---------------- 1. Check the input column ----------------
    required_cols = ["No", "Longitude", "Latitude"]
    for col in required_cols:
        if col not in sinkhole_position.columns:
            raise ValueError(
                f"[LeafAreaIndex] The input sinkhole_position is missing a required column: '{col}'"
            )

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
            f"[LeafAreaIndex] Unsupported SSP scenarios:{ssp}, currently only supports{list(ssp_to_sspxxx.keys())}"
        )

    sspxxx = ssp_to_sspxxx[ssp_key]  # Such as 'ssp3' -> 'ssp370'

    # ---------------- 3. Construct the data path root directory ----------------
    lai_root = os.path.join(
        database_folder_path,
        "10_leaf_area_1983_2100_0.05d",
    )

    # History: ...\\historical\\lai_historical_2xxx_y.tif
    hist_dir = os.path.join(lai_root, "historical")

    # :...\\sspxxx\\lai_sspxxx_2xxx_y.tif
    # Note: Files and folders ignore case. Lowercase is used here.
    ssp_dir = os.path.join(lai_root, sspxxx)

    print(f"[LeafAreaIndex] Historical data directory:{hist_dir}")
    print(f"[LeafAreaIndex] SSP :{ssp_dir}")
    print(f"[LeafAreaIndex] Current SSP scenario:{ssp} -> {sspxxx}")

    # ---------------- 4. ----------------
    # History: 2000, 2010, 2020
    hist_years = [2000, 2010, 2020]

    # Future time window: 2020-2040; 2040-2060; 2060-2080; 2080-2100
    future_windows = {
        "LAI_2020_2040": [2020, 2030, 2040],
        "LAI_2040_2060": [2040, 2050, 2060],
        "LAI_2060_2080": [2060, 2070, 2080],
        "LAI_2080_2100": [2080, 2090, 2100],
    }

    all_future_years = sorted({y for years in future_windows.values() for y in years})
    years_needed = sorted(set(hist_years + all_future_years))

    # Representative months of each year (4, 8, 12)
    rep_months = [4, 8, 12]

    # ---------------- 5. Calculated by "year": each year = the average of three representative months ----------------
    year_values = {}

    for year in years_needed:
        month_arrays = []
        for month in rep_months:
            # SSP
            if year < 2015:
                tif_name = f"lai_historical_{year}_{month}.tif"
                tif_path = os.path.join(hist_dir, tif_name)
            else:
                # Take the SSP path in 2015 and later: lai_sspxxx_2xxx_y.tif
                # Such as: lai_ssp370_2020_4.tif
                tif_name = f"lai_{sspxxx}_{year}_{month}.tif"
                tif_path = os.path.join(ssp_dir, tif_name)

            label = f"{year}-M{month}"
            print(f"[LeafAreaIndex] year{year}month{month}Raster used:{tif_path}")
            arr = _extract_lai_values_for_month(tif_path, sinkhole_position, label=label)
            month_arrays.append(arr)

        # LAI = 4/8/12
        stack = np.vstack(month_arrays)
        year_mean = np.nanmean(stack, axis=0)
        year_values[year] = year_mean

    # ---------------- 6. History: Annual average in 2000, 2010, 2020 ----------------
    print("[LeafAreaIndex] Calculate the historical average (average annual LAI in 2000, 2010, 2020)...")
    hist_stack = np.vstack([year_values[y] for y in hist_years])
    hist_mean = np.nanmean(hist_stack, axis=0)
    hist_col = "LAI_hist_2000_2010_2020"
    sinkhole_position[hist_col] = hist_mean

    # ---------------- 7. Average in each future time period ----------------
    for col_name, years in future_windows.items():
        print(f"[LeafAreaIndex] Calculate the future time period{col_name}Corresponding year{years}LAI ...")
        stack = np.vstack([year_values[y] for y in years])
        mean_vals = np.nanmean(stack, axis=0)
        sinkhole_position[col_name] = mean_vals

    # ---------------- 8. Save historical and future results ----------------
    # :ID + +
    hist_out_cols = ["No", "Longitude", "Latitude", hist_col]
    hist_out_cols = [c for c in hist_out_cols if c in sinkhole_position.columns]
    hist_df = sinkhole_position[hist_out_cols].copy()

    os.makedirs(historical_folder_path, exist_ok=True)
    hist_output_path = os.path.join(
        historical_folder_path, "LeafAreaIndex_historical_2000_2010_2020.csv"
    )
    hist_df.to_csv(hist_output_path, index=False, encoding="utf-8-sig")

    # Future results: ID + coordinates + average of four time periods
    future_out_cols = ["No", "Longitude", "Latitude"] + list(future_windows.keys())
    future_out_cols = [c for c in future_out_cols if c in sinkhole_position.columns]
    future_df = sinkhole_position[future_out_cols].copy()

    os.makedirs(future_ssp_folder_path, exist_ok=True)
    future_output_path = os.path.join(
        future_ssp_folder_path, f"LeafAreaIndex_future_{ssp}.csv"
    )
    future_df.to_csv(future_output_path, index=False, encoding="utf-8-sig")

    # ---------------- 9. Print statistical information ----------------
    print("\\n[LeafAreaIndex] (2000, 2010, 2020 LAI ):")
    if not hist_df[hist_col].isna().all():
        print(f"Points:{len(hist_df)}")
        print(f"Minimum value:{hist_df[hist_col].min():.4f}")
        print(f":{hist_df[hist_col].max():.4f}")
        print(f"Average:{hist_df[hist_col].mean():.4f}")

    print("\\n[LeafAreaIndex] ( SSP ):")
    for col_name in future_windows.keys():
        col_series = future_df[col_name]
        if col_series.isna().all():
            continue
        print(f"  {col_name}:")
        print(f":{len(col_series)}")
        print(f":{col_series.min():.4f}")
        print(f"Maximum value:{col_series.max():.4f}")
        print(f"Average:{col_series.mean():.4f}")

    print("\\n[LeafAreaIndex] Historical results have been saved to:")
    print("  ", hist_output_path)
    print("[LeafAreaIndex] Future results saved to:")
    print("  ", future_output_path)
    print("\\n[LeafAreaIndex] Result preview (history part):")
    print(hist_df.head())
    print("\\n[LeafAreaIndex] Result preview (future part):")
    print(future_df.head())

    return sinkhole_position
