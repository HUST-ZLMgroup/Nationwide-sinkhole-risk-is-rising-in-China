# impervious_index.py
import os
import numpy as np
from rasterio_compat import rasterio
from tqdm import tqdm


def _extract_isa_values_for_year(tif_path, sinkhole_position, year_label=None):
    """GeoTIFF , sinkhole_position numpy ."""
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"[ImperviousIndex] Raster file not found:{tif_path}")

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


def impervious_index(
    sinkhole_position,
    database_folder_path,
    historical_folder_path,
    future_ssp_folder_path,
    ssp,
):
    """Extract the impermeability index (0-100, 0 is permeable, 100 is impermeable) and calculate the historical/future time period average.

    data path
    --------
    - 2015 and before (history):
      Z:\\jing\\Large_scale\\future_dataset\\8_imp ervious_surface_area_1985_2100_1km\\historical\\gUrban_ISA_1km_2xxx.tif

    - 2020-2100 (SSP projection):
      Z:\\jing\\Large_scale\\future_dataset\\8_i mpervious_surface_area_1985_2100_1km\\SSPx\\SSPx_gISA_2xxx_1km.tif
      Where SSPx is the uppercase SSP name (SSP1/SSP2/...), 2xxx is the year (2020-2100, 5-year interval)

    Computing requirements
    --------
    - History: Use the three-year average of 2000, 2010, 2020
      (Note: 2000, 2010 are from historical, 2020 is from SSP path)
    - Future: Using the average of 10-year intervals over a 20-year period:
        * 2020-2040: Use 2020, 2030, 2040
        * 2040-2060: Use 2040, 2050, 2060
        * 2060-2080: Use 2060, 2070, 2080
        * 2080-2100: Use 2080, 2090, 2100

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
        SSP context string, such as 'ssp1', 'ssp2', 'ssp3', 'ssp4', 'ssp5'

    Return
    ----
    pandas.DataFrame
        Add the following columns to the original DataFrame:
        -ImperviousIndex_hist_2000_2010_2020
        -ImperviousIndex_2020_2040
        -ImperviousIndex_2040_2060
        -ImperviousIndex_2060_2080
        -ImperviousIndex_2080_2100"""

    print("\\n[ImperviousIndex] Start calculating the impermeability index...")

    # ---------------- 1. Check the input column ----------------
    required_cols = ["No", "Longitude", "Latitude"]
    for col in required_cols:
        if col not in sinkhole_position.columns:
            raise ValueError(
                f"[ImperviousIndex] The input sinkhole_position is missing a required column: '{col}'"
            )

    # ---------------- 2. Construct the data path root directory ----------------
    isa_root = os.path.join(
        database_folder_path,
        "8_impervious_surface_area_1985_2100_1km",
    )

    # Historical path: .../historical/gUrban_ISA_1km_2xxx.tif
    hist_dir = os.path.join(isa_root, "historical")

    # SSP path:.../SSPx/SSPx_gISA_2xxx_1km.tif
    ssp_folder_name = ssp.upper()  # 'ssp3' -> 'SSP3'
    ssp_dir = os.path.join(isa_root, ssp_folder_name)

    print(f"[ImperviousIndex] Historical data directory:{hist_dir}")
    print(f"[ImperviousIndex] SSP :{ssp_dir}")
    print(f"[ImperviousIndex] SSP :{ssp}")

    # ---------------- 3. ----------------
    # History: 2000, 2010, 2020
    hist_years = [2000, 2010, 2020]

    # Future time window: 2020-2040; 2040-2060; 2060-2080; 2080-2100
    future_windows = {
        "ImperviousIndex_2020_2040": [2020, 2030, 2040],
        "ImperviousIndex_2040_2060": [2040, 2050, 2060],
        "ImperviousIndex_2060_2080": [2060, 2070, 2080],
        "ImperviousIndex_2080_2100": [2080, 2090, 2100],
    }

    all_future_years = sorted({y for years in future_windows.values() for y in years})
    years_needed = sorted(set(hist_years + all_future_years))

    # ---------------- 4. Splice the correct path for each year and extract the value ----------------
    year_values = {}

    for year in years_needed:
        if year <= 2015:
            # Processing step.
            tif_name = f"gUrban_ISA_1km_{year}.tif"
            tif_path = os.path.join(hist_dir, tif_name)
        else:
            # SSP path (2020-2100, in 5-year increments, but we only use years spaced 10 years apart)
            # File name format: SSPx_gISA_2xxx_1km.tif, for example: SSP3_gISA_2020_1km.tif
            tif_name = f"{ssp_folder_name}_gISA_{year}_1km.tif"
            tif_path = os.path.join(ssp_dir, tif_name)

        print(f"[ImperviousIndex] Year{year}Raster used:{tif_path}")
        year_values[year] = _extract_isa_values_for_year(
            tif_path, sinkhole_position, year_label=str(year)
        )

    # ---------------- 5. History: 2000, 2010, 2020 Average ----------------
    print("[ImperviousIndex] Calculating the historical average (2000, 2010, 2020)...")
    hist_stack = np.vstack([year_values[y] for y in hist_years])
    hist_mean = np.nanmean(hist_stack, axis=0)
    hist_col = "ImperviousIndex_hist_2000_2010_2020"
    sinkhole_position[hist_col] = hist_mean

    # ---------------- 6. Average in the next four time periods ----------------
    for col_name, years in future_windows.items():
        print(f"[ImperviousIndex] Calculate the future time period{col_name}Corresponding year{years}...")
        stack = np.vstack([year_values[y] for y in years])
        mean_vals = np.nanmean(stack, axis=0)
        sinkhole_position[col_name] = mean_vals

    # ---------------- 7. ----------------
    # :ID + +
    hist_out_cols = ["No", "Longitude", "Latitude", hist_col]
    hist_out_cols = [c for c in hist_out_cols if c in sinkhole_position.columns]
    hist_df = sinkhole_position[hist_out_cols].copy()

    os.makedirs(historical_folder_path, exist_ok=True)
    hist_output_path = os.path.join(
        historical_folder_path, "ImperviousIndex_historical_2000_2010_2020.csv"
    )
    hist_df.to_csv(hist_output_path, index=False, encoding="utf-8-sig")

    # Future results: ID + coordinates + average of four time periods
    future_out_cols = ["No", "Longitude", "Latitude"] + list(future_windows.keys())
    future_out_cols = [c for c in future_out_cols if c in sinkhole_position.columns]
    future_df = sinkhole_position[future_out_cols].copy()

    os.makedirs(future_ssp_folder_path, exist_ok=True)
    future_output_path = os.path.join(
        future_ssp_folder_path, f"ImperviousIndex_future_{ssp}.csv"
    )
    future_df.to_csv(future_output_path, index=False, encoding="utf-8-sig")

    # ---------------- 8. ----------------
    print("\\n[ImperviousIndex] historical data statistics (2000, 2010, 2020 average):")
    if not hist_df[hist_col].isna().all():
        print(f"Points:{len(hist_df)}")
        print(f"Minimum value:{hist_df[hist_col].min():.2f}")
        print(f":{hist_df[hist_col].max():.2f}")
        print(f"Average:{hist_df[hist_col].mean():.2f}")

    print("\\n[ImperviousIndex] ( SSP ):")
    for col_name in future_windows.keys():
        col_series = future_df[col_name]
        if col_series.isna().all():
            continue
        print(f"  {col_name}:")
        print(f":{len(col_series)}")
        print(f":{col_series.min():.2f}")
        print(f"Maximum value:{col_series.max():.2f}")
        print(f"Average:{col_series.mean():.2f}")

    print("\\n[ImperviousIndex] Historical results have been saved to:")
    print("  ", hist_output_path)
    print("[ImperviousIndex] Future results saved to:")
    print("  ", future_output_path)
    print("\\n[ImperviousIndex] ():")
    print(hist_df.head())
    print("\\n[ImperviousIndex] ():")
    print(future_df.head())

    return sinkhole_position
