# distance_to_karst.py
import os
import geopandas as gpd
from shapely.geometry import Point


def distance_to_karst(sinkhole_position, database_folder_path, output_folder_path):
    """Calculate the distance from each collapse point to the nearest carbonate karst zone and save the results as a CSV.

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
        Add the object after the 'Distance_to_karst' column based on the original DataFrame."""

    print("\\n[Distance_to_karst] Start calculating the shortest distance from the collapse point to the karst area...")

    # ---------------- 1. Check the input column ----------------
    required_cols = ["No", "Longitude", "Latitude"]
    for col in required_cols:
        if col not in sinkhole_position.columns:
            raise ValueError(
                f"[Distance_to_karst] sinkhole_position : '{col}'"
            )

    # ---------------- 2. Read karst area vector data ----------------
    karst_path = os.path.join(
        database_folder_path,
        "Geological_factors",
        "Karst_region",
        "Karst distribution in China_ExportFeatures.shp",
    )
    print(f"[Distance_to_karst] Read karst area data:{karst_path}")
    karst_gdf = gpd.read_file(karst_path)

    if "rock_type" not in karst_gdf.columns:
        raise ValueError(
            "[Distance_to_karst] Field 'rock_type' not found in karst area data, unable to filter carbonate rock areas."
        )

    # 3. (rock_type 1 2)
    carbonate_karst = karst_gdf[karst_gdf["rock_type"].isin([1, 2])].copy()
    if carbonate_karst.empty:
        raise ValueError(
            "[Distance_to_karst] The carbonate rock area is empty after filtering, please check the rock_type field or filter condition."
        )

    print(
        f"[Distance_to_karst] Number of carbonate polygons:{len(carbonate_karst)}, "
        f"Total number of elements:{len(karst_gdf)}"
    )

    # ---------------- 4. GeoDataFrame ----------------
    sinkhole_df = sinkhole_position.copy()
    geometry = [
        Point(lon, lat)
        for lon, lat in zip(sinkhole_df["Longitude"], sinkhole_df["Latitude"])
    ]
    sinkhole_gdf = gpd.GeoDataFrame(sinkhole_df, geometry=geometry, crs="EPSG:4326")

    # ---------------- 5. & ----------------
    # karst CRS(WGS84,)
    if carbonate_karst.crs is None:
        raise ValueError(
            "[Distance_to_karst] Karst area data lacks CRS information, please define the coordinate system in shp."
        )

    if carbonate_karst.crs != sinkhole_gdf.crs:
        carbonate_karst = carbonate_karst.to_crs(sinkhole_gdf.crs)

    # Then switch to the equidistant projected coordinate system (UTM or Web Mercator), with meters as the distance unit
    try:
        utm_crs = sinkhole_gdf.estimate_utm_crs()
        print(f"[Distance_to_karst] UTM :{utm_crs}")
        sinkhole_gdf_utm = sinkhole_gdf.to_crs(utm_crs)
        carbonate_karst_utm = carbonate_karst.to_crs(utm_crs)
    except Exception as e:
        print(
            "[Distance_to_karst] UTM , EPSG:3857 (Web Mercator)."
        )
        print(f"[Distance_to_karst] Specific error information:{e}")
        sinkhole_gdf_utm = sinkhole_gdf.to_crs("EPSG:3857")
        carbonate_karst_utm = carbonate_karst.to_crs("EPSG:3857")

    # ---------------- 6. Calculate the nearest distance (unit: meters) ----------------
    print("[Distance_to_karst] Calculate the nearest distance from the point to the carbonate rock area...")

    # Combine all carbonate rock areas into one geometry to reduce calculations
    karst_union = carbonate_karst_utm.geometry.unary_union

    # Processing step.
    distances = sinkhole_gdf_utm.geometry.apply(lambda geom: geom.distance(karst_union))

    # ---------------- 7. Write the distance back to the original DataFrame ----------------
    sinkhole_position["Distance_to_karst"] = distances.values

    # ---------------- 8. Organize and save the results ----------------
    # Only necessary columns are output here. If you need other columns later, you can merge them in the main code yourself.
    result_columns = ["No", "Longitude", "Latitude", "Distance_to_karst"]
    result_columns = [c for c in result_columns if c in sinkhole_position.columns]
    final_result = sinkhole_position[result_columns].copy()

    output_path = os.path.join(output_folder_path, "Distance_to_karst.csv")
    final_result.to_csv(output_path, index=False, encoding="utf-8-sig")

    # ---------------- 9. Print statistical information ----------------
    print("\\n[Distance_to_karst] processing completed! Results have been saved to:")
    print(" ", output_path)
    print(f"[Distance_to_karst] Total collapse points:{len(final_result)}")

    if len(final_result) > 0:
        print("\\n[Distance_to_karst] Distance statistics (meters):")
        print(f"Minimum distance:{final_result['Distance_to_karst'].min():.2f}")
        print(f"Maximum distance:{final_result['Distance_to_karst'].max():.2f}")
        print(f"Average distance:{final_result['Distance_to_karst'].mean():.2f}")

        print("\\n[Distance_to_karst] :")
        print(final_result.head())

    return sinkhole_position
