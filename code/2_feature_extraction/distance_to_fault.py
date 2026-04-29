# distance_to_fault.py
import os
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm


def distance_to_fault(sinkhole_position, database_folder_path, output_folder_path):
    """Calculate the distance (in meters) from each collapse point to the nearest fault and save the results as a CSV file.

    parameters
    ----
    sinkhole_position : pandas.DataFrame
        The DataFrame that has been extracted in the main code needs to contain at least columns:
        'No', 'Longitude', 'Latitude'
    database_folder_path : str
        The database root directory defined in the main code, for example:
        Z:\\jing\\Large_scale\\future_dataset
    output_folder_path : str
        The output folder path mapped by the main code based on df_path.

    Return
    ----
    pandas.DataFrame
        Add the object after the 'Distance_to_Fault_m' column based on the original DataFrame."""

    print("\\n[Distance_to_fault] Start calculating the nearest distance from the collapse point to the fault...")

    # ---------------- 1. Check the input column ----------------
    required_cols = ["No", "Longitude", "Latitude"]
    for col in required_cols:
        if col not in sinkhole_position.columns:
            raise ValueError(
                f"[Distance_to_fault] The input sinkhole_position is missing a required column: '{col}'"
            )

    # ---------------- 2. Set the fault data path and read ----------------
    fault_path = os.path.join(
        database_folder_path,
        "Geological_factors",
        "Fault",
        "outputs",
        "faults.shp",
    )
    if not os.path.exists(fault_path):
        raise FileNotFoundError(
            f"[Distance_to_fault] Fault vector file not found:{fault_path}"
        )

    print(f"[Distance_to_fault] Read fault data:{fault_path}")
    fault_gdf = gpd.read_file(fault_path)

    # ---------------- 3. Create a collapse point GeoDataFrame ----------------
    sinkhole_df = sinkhole_position.copy()
    geometry = [
        Point(lon, lat)
        for lon, lat in zip(sinkhole_df["Longitude"], sinkhole_df["Latitude"])
    ]
    sinkhole_gdf = gpd.GeoDataFrame(sinkhole_df, geometry=geometry, crs="EPSG:4326")

    # ---------------- 4. Coordinate system for processing tomographic data ----------------
    if fault_gdf.crs is None:
        print(
            "[Distance_to_fault] Warning: Fault data has no defined coordinate system, manually set to EPSG:4326 (WGS84)"
        )
        fault_gdf.crs = "EPSG:4326"

    # If the CRS of the fault and the point are inconsistent, first unify to WGS84
    if fault_gdf.crs != sinkhole_gdf.crs:
        print(
            f"[Distance_to_fault] Change fault data from{fault_gdf.crs}Reproject to EPSG:4326 (WGS84)..."
        )
        fault_gdf = fault_gdf.to_crs(sinkhole_gdf.crs)

    # ---------------- 5. Project to the equidistant projection coordinate system (meters) ----------------
    # :EPSG:4479 - China Albers Equal Area Conic
    projected_crs = "EPSG:4479"
    print(f"[Distance_to_fault] Project fault and collapse point data to{projected_crs} ...")

    fault_gdf_projected = fault_gdf.to_crs(projected_crs)
    sinkhole_gdf_projected = sinkhole_gdf.to_crs(projected_crs)

    # ---------------- 6. () ----------------
    print("[Distance_to_fault] Start calculating the distance to the nearest fault (meters)...")

    # Combine all fault geometries to reduce distance calculations
    fault_union = fault_gdf_projected.geometry.unary_union

    distances_meters = []
    for geom in tqdm(
        sinkhole_gdf_projected.geometry,
        total=len(sinkhole_gdf_projected),
        desc="Calculate the distance to the nearest fault",
    ):
        try:
            dist_val = geom.distance(fault_union)
            distances_meters.append(float(dist_val))
        except Exception as e:
            # Log NaN on error
            distances_meters.append(np.nan)

    if len(distances_meters) != len(sinkhole_position):
        raise RuntimeError(
            "[Distance_to_fault] The number of calculated distances is inconsistent with the number of input points."
            f"points={len(sinkhole_position)}, distances={len(distances_meters)}."
        )

    # ---------------- 7. Write the distance back to the DataFrame ----------------
    sinkhole_position["Distance_to_Fault_m"] = distances_meters

    # ---------------- 8. Save the results ----------------
    result_cols = ["No", "Longitude", "Latitude", "Distance_to_Fault_m"]
    result_cols = [c for c in result_cols if c in sinkhole_position.columns]
    result_df = sinkhole_position[result_cols].copy()

    output_path = os.path.join(output_folder_path, "Distance_to_Fault.csv")
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # ---------------- 9. Print statistical information ----------------
    print("\\n[Distance_to_fault] processing completed! Results have been saved to:")
    print(" ", output_path)
    print(f"[Distance_to_fault] Total collapse points:{len(result_df)}")

    if result_df["Distance_to_Fault_m"].notna().any():
        print(
            f"[Distance_to_fault] :"
            f"{result_df['Distance_to_Fault_m'].mean():.2f}m"
        )
        print(
            f"[Distance_to_fault] Minimum distance to fault:"
            f"{result_df['Distance_to_Fault_m'].min():.2f}m"
        )
        print(
            f"[Distance_to_fault] :"
            f"{result_df['Distance_to_Fault_m'].max():.2f}m"
        )

    print("\\n[Distance_to_fault] :")
    print(result_df.head())

    return sinkhole_position
