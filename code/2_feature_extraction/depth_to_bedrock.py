# depth_to_bedrock.py
import os
import numpy as np
from rasterio_compat import rasterio
from tqdm import tqdm


def depth_to_bedrock(sinkhole_position, database_folder_path, output_folder_path):
    """Extract the bedrock depth (Depth_to_Bedrock) for each collapse point and save it as a CSV file.

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
        Add the object after the 'Depth_to_Bedrock' column based on the original DataFrame."""

    print("\\n[Depth_to_bedrock] Start extracting bedrock depth information...")

    # ---------------- 1. Check the input column ----------------
    required_cols = ["No", "Longitude", "Latitude"]
    for col in required_cols:
        if col not in sinkhole_position.columns:
            raise ValueError(
                f"[Depth_to_bedrock] sinkhole_position : '{col}'"
            )

    # ---------------- 2. Set bedrock depth TIFF path ----------------
    dtb_path = os.path.join(
        database_folder_path,
        "Geological_factors",
        "Depth_to_bedrock",
        "DTB_CHINA_1k.tif",
    )
    if not os.path.exists(dtb_path):
        raise FileNotFoundError(
            f"[Depth_to_bedrock] Bedrock depth raster file not found:{dtb_path}"
        )

    print(f"[Depth_to_bedrock] Read bedrock depth raster:{dtb_path}")

    dtb_values = []

    # ---------------- 3. Use rasterio to open TIFF and extract point by point ----------------
    with rasterio.open(dtb_path) as src:
        print(f"[Depth_to_bedrock] Raster CRS:{src.crs}")
        print(
            f"[Depth_to_bedrock] : ={src.width}, ={src.height}, resolution={src.res}"
        )

        # tqdm progress bar
        for idx, row in tqdm(
            sinkhole_position.iterrows(),
            total=len(sinkhole_position),
            desc="Extract bedrock depth",
        ):
            lon = row["Longitude"]
            lat = row["Latitude"]

            try:
                # Processing step.
                # Note: src.index(x, y) returns (row, col)
                row_idx, col_idx = src.index(lon, lat)
            except Exception:
                # ,
                dtb_values.append(np.nan)
                continue

            # Determine whether it is within the grid range
            if 0 <= row_idx < src.height and 0 <= col_idx < src.width:
                window = rasterio.windows.Window(col_idx, row_idx, 1, 1)
                value = src.read(1, window=window)
                dtb_values.append(float(value[0][0]))
            else:
                dtb_values.append(np.nan)

    # ---------------- 4. Write back DataFrame ----------------
    if len(dtb_values) != len(sinkhole_position):
        raise RuntimeError(
            "[Depth_to_bedrock] The number of extracted bedrock depths is inconsistent with the number of input points."
            f"points={len(sinkhole_position)}, values={len(dtb_values)}."
        )

    sinkhole_position["Depth_to_Bedrock"] = dtb_values

    # ---------------- 5. Save the results ----------------
    output_path = os.path.join(output_folder_path, "Depth_to_Bedrock.csv")
    sinkhole_position.to_csv(output_path, index=False, encoding="utf-8-sig")

    # ---------------- 6. Print statistical information ----------------
    print("\\n[Depth_to_bedrock] processing completed! Results have been saved to:")
    print(" ", output_path)
    print(f"[Depth_to_bedrock] Total collapse points:{len(sinkhole_position)}")
    print(
        f"[Depth_to_bedrock] The number of points successfully extracted from the depth:"
        f"{sinkhole_position['Depth_to_Bedrock'].notna().sum()}"
    )

    if sinkhole_position["Depth_to_Bedrock"].notna().any():
        mean_val = sinkhole_position["Depth_to_Bedrock"].mean()
        print(f"[Depth_to_bedrock] Average bedrock depth:{mean_val:.2f}m")

    print("\\n[Depth_to_bedrock] Result preview:")
    print(sinkhole_position.head())

    return sinkhole_position
