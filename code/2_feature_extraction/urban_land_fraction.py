# urban_land_fraction.py
import os
import numpy as np
from rasterio_compat import rasterio
from tqdm import tqdm


def _extract_raster_values_for_year(tif_path, sinkhole_position, year_label=None):
    """Extracts cell values point by point from the given GeoTIFF, returning a numpy array with the same length as sinkhole_position."""
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"[Urban_land_fraction] Raster file not found:{tif_path}")

    desc = f"Extract {year_label}" if year_label is not None else f"Extract {os.path.basename(tif_path)}"
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


def urban_land_fraction(
    sinkhole_position,
    database_folder_path,
    historical_folder_path,
    future_ssp_folder_path,
    ssp,
):
    """Calculate the ratio of urban land to all land area (urban land fraction): - History: average of three moments in 2000, 2010, 2020; - Future: average over four time periods (10-year interval data): 2020-2040, 2040-2060, 2060-2080, 2080-2100. parameters ---- sinkhole_position : pandas.DataFrame Contains at least the columns 'No', 'Longitude', 'Latitude' database_folder_path : str Database root directory, for example Z:\\jing\\Large_scale\\future_dataset historical_folder_path : str Historical data output directory (.../historical) future_ssp_folder_path : str (.../future/sspX) ssp : str SSP context string, such as 'ssp1', 'ssp2', 'ssp3', 'ssp4', 'ssp5' Return ---- pandas.DataFrame Add the following columns to the original DataFrame: - UrbanFrac_hist_2000_2010_2020 - UrbanFrac_2020_2040 - UrbanFrac_2040_2060 - UrbanFrac_2060_2080 - UrbanFrac_2080_2100"""

    print("\\n[Urban_land_fraction] Start calculating the ratio of urban land to all land area...")

    # ---------------- 1. Check the input column ----------------
    required_cols = ["No", "Longitude", "Latitude"]
    for col in required_cols:
        if col not in sinkhole_position.columns:
            raise ValueError(
                f"[Urban_land_fraction] The input sinkhole_position is missing a required column: '{col}'"
            )

    # ---------------- 2. Construct data path ----------------
    # Root directory: 3_fraction_of_urban_land_2000_2100_0.125d/UrbanFraction_1_8_dgr_GEOTIFF_v1
    urban_root = os.path.join(
        database_folder_path,
        "3_fraction_of_urban_land_2000_2100_0.125d",
        "UrbanFraction_1_8_dgr_GEOTIFF_v1",
    )

    # 2000 base year data
    base2000_dir = os.path.join(
        urban_root,
        "UrbanFraction_1_8_dgr_GEOTIFF_BaseYear_2000_v1",
    )
    base2000_path = os.path.join(base2000_dir, "urb_frac_2000.tif")

    # SSP projection data 2010-2100
    proj_dir = os.path.join(
        urban_root,
        "UrbanFraction_1_8_dgr_GEOTIFF_Projections_SSPs1-5_2010-2100_v1",
    )

    print(f"[Urban_land_fraction] BaseYear 2000 Raster path:{base2000_path}")
    print(f"[Urban_land_fraction] SSP projection data directory:{proj_dir}")
    print(f"[Urban_land_fraction] Current SSP scenario:{ssp}")

    # ---------------- 3. ----------------
    # History: 2000, 2010, 2020
    hist_years = [2000, 2010, 2020]

    # Future time window: 2020-2040; 2040-2060; 2060-2080; 2080-2100
    # 10 ,
    future_windows = {
        "UrbanFrac_2020_2040": [2020, 2030, 2040],
        "UrbanFrac_2040_2060": [2040, 2050, 2060],
        "UrbanFrac_2060_2080": [2060, 2070, 2080],
        "UrbanFrac_2080_2100": [2080, 2090, 2100],
    }

    all_future_years = sorted({y for years in future_windows.values() for y in years})
    years_needed = sorted(set(hist_years + all_future_years))

    # ---------------- 4. Extract raster values for each year ----------------
    year_values = {}

    for year in years_needed:
        if year == 2000:
            tif_path = base2000_path
        else:
            # 2010-2100: Use the projection file corresponding to SSP, naming format: sspx_2xxx.tif
            # ssp3_2010.tif, ssp3_2020.tif ...
            tif_name = f"{ssp}_{year}.tif"
            tif_path = os.path.join(proj_dir, tif_name)

        print(f"[Urban_land_fraction] Year{year}Raster used:{tif_path}")
        year_values[year] = _extract_raster_values_for_year(
            tif_path, sinkhole_position, year_label=str(year)
        )

    # ---------------- 5. History: 2000, 2010, 2020 Average ----------------
    print("[Urban_land_fraction] Calculate historical average (2000, 2010, 2020)...")
    hist_stack = np.vstack([year_values[y] for y in hist_years])
    hist_mean = np.nanmean(hist_stack, axis=0)
    hist_col = "UrbanFrac_hist_2000_2010_2020"
    sinkhole_position[hist_col] = hist_mean

    # ---------------- 6. Average in the next four time periods ----------------
    for col_name, years in future_windows.items():
        print(f"[Urban_land_fraction]{col_name}Corresponding year{years}...")
        stack = np.vstack([year_values[y] for y in years])
        mean_vals = np.nanmean(stack, axis=0)
        sinkhole_position[col_name] = mean_vals

    # ---------------- 7. ----------------
    # : ID + +
    hist_out_cols = ["No", "Longitude", "Latitude", hist_col]
    hist_out_cols = [c for c in hist_out_cols if c in sinkhole_position.columns]
    hist_df = sinkhole_position[hist_out_cols].copy()

    os.makedirs(historical_folder_path, exist_ok=True)
    hist_output_path = os.path.join(
        historical_folder_path, "UrbanFraction_historical_2000_2010_2020.csv"
    )
    hist_df.to_csv(hist_output_path, index=False, encoding="utf-8-sig")

    # Future results: ID + coordinate + average of four time periods
    future_out_cols = ["No", "Longitude", "Latitude"] + list(future_windows.keys())
    future_out_cols = [c for c in future_out_cols if c in sinkhole_position.columns]
    future_df = sinkhole_position[future_out_cols].copy()

    os.makedirs(future_ssp_folder_path, exist_ok=True)
    future_output_path = os.path.join(
        future_ssp_folder_path, f"UrbanFraction_future_{ssp}.csv"
    )
    future_df.to_csv(future_output_path, index=False, encoding="utf-8-sig")

    # ---------------- 8. ----------------
    print("\\n[Urban_land_fraction] Historical data statistics (2000, 2010, 2020 average):")
    if not hist_df[hist_col].isna().all():
        print(f"Points:{len(hist_df)}")
        print(f"Minimum value:{hist_df[hist_col].min():.4f}")
        print(f":{hist_df[hist_col].max():.4f}")
        print(f"Average:{hist_df[hist_col].mean():.4f}")

    print("\\n[Urban_land_fraction] Statistics for each future time period (according to SSP scenario):")
    for col_name in future_windows.keys():
        col_series = future_df[col_name]
        if col_series.isna().all():
            continue
        print(f"  {col_name}:")
        print(f":{len(col_series)}")
        print(f":{col_series.min():.4f}")
        print(f"Maximum value:{col_series.max():.4f}")
        print(f"Average:{col_series.mean():.4f}")

    print("\\n[Urban_land_fraction] Historical results have been saved to:")
    print("  ", hist_output_path)
    print("[Urban_land_fraction] Future results saved to:")
    print("  ", future_output_path)
    print("\\n[Urban_land_fraction] Result preview (history part):")
    print(hist_df.head())
    print("\\n[Urban_land_fraction] Result preview (future part):")
    print(future_df.head())

    return sinkhole_position
