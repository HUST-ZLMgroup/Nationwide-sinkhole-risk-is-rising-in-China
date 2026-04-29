# aggregate_features.py
import os
import pandas as pd


def aggregate_features(
    sinkhole_position,
    df_path,
    output_folder_path,
    ssp=None,
):
    """Summarize all extracted features and save as a summary CSV.

    Assumptions:
    ----
    - Requirements 1~10 have been completed in sequence, and feature columns are continuously added to the same sinkhole_position;
    - sinkhole_position contains at least the original columns:
        ['No', 'Disaster', 'Longitude', 'Latitude', 'ADCODE99', 'Province']

    parameters
    ----
    sinkhole_position : pandas.DataFrame
        Total DataFrame that contains all feature columns.
    df_path : str
        The original CSV path read in the main code is used to extract the file name from it and spell it into the output file name.
    output_folder_path : str
        The directory mapped by the main code according to csv_to_subfolder:
        D:\\path\\to\\sinkhole-risk-china\\data\\Extracted_HAVE_future\\<csv_to_subfolder[csv_name]>
        This function will save the summary results in this directory.
    ssp: str, optional
        The SSP context currently in use (e.g. 'ssp1'/'ssp2'/'ssp3'/'ssp5'),
        If provided, it will be spelled into the output file name to distinguish different scenarios.

    Return
    ----
    pandas.DataFrame
        Aggregated DataFrame (same as input sinkhole_position, just column order rearranged)."""

    print("\\n[AggregateFeatures] Start aggregating all features into a single file...")

    # 1. Check whether the original necessary column exists
    # base_cols = ["No", "Disaster", "Longitude", "Latitude", "ADCODE99", "Province"]
    base_cols = ["No", "Longitude", "Latitude"]
    missing = [c for c in base_cols if c not in sinkhole_position.columns]
    if missing:
        raise ValueError(
            f"[AggregateFeatures] sinkhole_position is missing necessary original columns:{missing}"
        )

    # 2. Arrange the column order: the original 6 columns are in the front, and the remaining features are in the back.
    all_cols = list(sinkhole_position.columns)
    other_cols = [c for c in all_cols if c not in base_cols]
    ordered_cols = base_cols + other_cols

    sinkhole_ordered = sinkhole_position[ordered_cols].copy()

    # 3.
    csv_name = os.path.basename(df_path)              # Positive_Negative_balanced_25366.csv
    csv_stem, _ = os.path.splitext(csv_name)          # -> Positive_Negative_balanced_25366

    # File name format:
    # AllFeatures_<original CSV name>.csv
    # If ssp is provided, it will be AllFeatures_<original CSV name>_<ssp>.csv
    if ssp is not None and str(ssp).strip() != "":
        ssp_str = str(ssp).replace(" ", "")
        out_filename = f"AllFeatures_{csv_stem}_{ssp_str}.csv"
    else:
        out_filename = f"AllFeatures_{csv_stem}.csv"

    os.makedirs(output_folder_path, exist_ok=True)
    output_path = os.path.join(output_folder_path, out_filename)

    # 4. Save CSV
    sinkhole_ordered.to_csv(output_path, index=False, encoding="utf-8-sig")

    # 5. Simple log
    print("\\n[AggregateFeatures] Aggregation completed! File saved to:")
    print("  ", output_path)
    print(f"[AggregateFeatures] Total number of samples:{len(sinkhole_ordered)}")
    print(f"[AggregateFeatures] Total number of features (including original columns):{len(sinkhole_ordered.columns)}")
    print("\\n[AggregateFeatures] List preview:")
    print(sinkhole_ordered.columns.tolist())
    print("\\n[AggregateFeatures] Data preview:")
    print(sinkhole_ordered.head())

    return sinkhole_ordered
