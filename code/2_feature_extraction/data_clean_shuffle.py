import os
import numpy as np
import pandas as pd


def _find_long_nan_runs(series, max_allowed=10):
    """Find consecutive NaN segments in series and return a list:
    [(start_pos, end_pos, length), ...], only segments with length > max_allowed are retained.
    The start_pos/end_pos here are based on the position index of the current Series (0, 1, 2, ...)."""
    is_na = series.isna().to_numpy()
    n = len(is_na)
    runs = []

    i = 0
    while i < n:
        if is_na[i]:
            start = i
            while i < n and is_na[i]:
                i += 1
            end = i - 1
            length = end - start + 1
            if length > max_allowed:
                runs.append((start, end, length))
        else:
            i += 1

    return runs


def _normalize_integer_column(series, col_name):
    """Unify the specified columns into integer encoding.

    Processing logic:
    1. Convert to numerical value first;
    2. Round non-integer values;
    3. Keep missing values and finally convert them to pandas nullable integer type Int64."""
    s_num = pd.to_numeric(series, errors="coerce")

    mask_valid = s_num.notna()
    mask_non_integer = mask_valid & (~np.isclose(s_num, np.round(s_num), atol=1e-8))
    n_non_integer = int(mask_non_integer.sum())

    if n_non_integer > 0:
        print(
            f"[DataClean]{col_name}There are{n_non_integer},."
        )

    s_rounded = s_num.round()
    return pd.Series(pd.array(s_rounded, dtype="Int64"), index=series.index)


def data_clean_and_shuffle(
    sinkhole_position,
    df_path,
    output_folder_path,
    ssp=None,
    random_state=42,
):
    """Requirement 12: Data cleaning + disruption

    Steps:
    1. For each feature column, delete the rows in the feature that exceed the range of mean ± 10*std;
    2. For each feature column, interpolate the rows where the null values are located;
       If there is a continuous NaN length > 10 in a column, specific information will be printed out (no error interrupt will be forced);
    3. Check whether there is a value in each feature column that accounts for > 10%, and randomly add or subtract very small numbers to these values;
    4. The data is scrambled as a whole (by row) and the index is reset;
    5. Save to output_folder_path, file name = file name of requirement 11 + "_cleaned".

    parameters
    ----
    sinkhole_position : pandas.DataFrame
        Requirement 11 The total DataFrame that has been summarized (contains the original 6 columns + all feature columns).
    df_path : str
        The original sample CSV path (df_path read in in the main code), used to construct the output filename.
    output_folder_path : str
        The path is the directory mapped by the main program according to csv_to_subfolder.
        That is: D:\\path\\to\\sinkhole-risk-china\\data\\Extracted_HAVE_future\\<csv_to_subfolder[csv_name]>
    ssp : str or None
        Current SSP context (e.g. 'ssp1', 'ssp2', 'ssp3', 'ssp5'). Need to be consistent with requirement 11.
    random_state : int or None
        Random seeds used to control random scrambling and perturbation for easy reproduction.

    Return
    ----
    pandas.DataFrame
        Cleaned + shuffled DataFrame."""

    print("\\n[DataClean] ...")

    # For repeatability, set a random seed
    if random_state is not None:
        np.random.seed(random_state)

    # 0. Make a copy to avoid modifying the original sinkhole_position reference
    df_clean = sinkhole_position.copy()

    # 0.1 The specification needs to maintain integer columns to avoid becoming non-integer due to floating point precision residues
    if "ADCODE99" in df_clean.columns:
        df_clean["ADCODE99"] = _normalize_integer_column(df_clean["ADCODE99"], "ADCODE99")
    if "Disaster" in df_clean.columns:
        df_clean["Disaster"] = _normalize_integer_column(df_clean["Disaster"], "Disaster")

    # 1. Define "original column" and "feature column"
    # base_cols = ["No", "Disaster", "Longitude", "Latitude", "ADCODE99", "Province"]
    base_cols = ["No", "Longitude", "Latitude"]
    if "ADCODE99" in df_clean.columns:
        base_cols.append("ADCODE99")
    if "Disaster" in df_clean.columns:
        base_cols.append("Disaster")

    for c in base_cols:
        if c not in df_clean.columns:
            raise ValueError(f"[DataClean] DataFrame missing base column:{c}")

    # Processing step.
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in base_cols]

    print(f"[DataClean] Number of numerical feature columns detected:{len(feature_cols)}")
    print("[DataClean] Feature column list:")
    print("  ", feature_cols)

    # 1) For each feature column, delete rows exceeding mean ± 10*std
    print("\\n[DataClean] Step 1: Remove extreme outliers (mean ± 10*std)...")
    for col in feature_cols:
        series = df_clean[col].astype("float64")

        mu = series.mean(skipna=True)
        sigma = series.std(skipna=True)

        if not np.isfinite(mu) or not np.isfinite(sigma) or sigma == 0:
            # The column is basically constant or all NaN, skip outlier deletion
            print(f"[DataClean] column{col}: std is invalid or 0, skips outlier detection.")
            continue

        lower = mu - 10.0 * sigma
        upper = mu + 10.0 * sigma

        mask_outlier = (series < lower) | (series > upper)
        n_outliers = int(mask_outlier.sum())

        if n_outliers > 0:
            print(
                f"[DataClean] column{col}:{n_outliers}items exceeded [{lower:.4g}, {upper:.4g}] outlier sample."
            )
            df_clean = df_clean.loc[~mask_outlier].copy()
        else:
            print(f"[DataClean] column{col}: No outliers.")

    # After deleting the row, reset the index
    df_clean.reset_index(drop=True, inplace=True)
    print(f"[DataClean] After outliers are deleted, the number of samples:{len(df_clean)}")

    # 2) Interpolate null values and check for consecutive NaN > 10
    print("\\n[DataClean] Step 2: Interpolate to fill missing values + check for long NaN segments...")
    for col in feature_cols:
        s = df_clean[col].astype("float64")
        if not s.isna().any():
            # Not missing, skip
            continue

        # 2.1 NaN 10
        long_runs = _find_long_nan_runs(s, max_allowed=10)
        if long_runs:
            print(f"[DataClean][]{col}There are more than 10 consecutive NaN segments:")
            for start, end, length in long_runs:
                print(
                    f"- Location interval: [{start}, {end}],:{length}(based on current DataFrame row order)"
                )

        # 2.2 ,()
        s_interp = s.interpolate(method="linear", limit_direction="both")
        df_clean[col] = s_interp

        # NaN( NaN )
        remaining_nan = df_clean[col].isna().sum()
        if remaining_nan > 0:
            print(
                f"[DataClean][Prompt] Column{col}After interpolation, there is still{remaining_nan}NaN,"
                f"."
            )

    # 3) Check whether each feature column has a value > 10%, and if so, add a small random perturbation
    print("\\n[DataClean] Step 3: Check and perturb duplicate values > 10%...")
    for col in feature_cols:
        s = df_clean[col].astype("float64")
        non_na = s.dropna()

        if len(non_na) == 0:
            print(f"[DataClean] column{col}: All are NaN, skip.")
            continue

        value_ratio = non_na.value_counts(normalize=True)
        # Select all values with proportion > 10%
        heavy_vals = value_ratio[value_ratio > 0.10]

        if heavy_vals.empty:
            print(f"[DataClean] column{col}: There is no single value accounting for >10%.")
            continue

        # Calculate a miniscule eps, relative to the scale of the column
        std_col = non_na.std(skipna=True)
        if not np.isfinite(std_col) or std_col == 0:
            base_scale = 1.0
        else:
            base_scale = std_col

        eps = base_scale * 1e-6
        if eps == 0:
            eps = 1e-6

        print(
            f"[DataClean] column{col}: exists{len(heavy_vals)}values account for >10%, random perturbations (±{eps:.3e})."
        )

        for val, ratio in heavy_vals.items():
            mask_val = s == val
            n_val = int(mask_val.sum())
            if n_val == 0:
                continue

            noise = (np.random.rand(n_val) - 0.5) * 2.0 * eps  # in [-eps, eps]
            s.loc[mask_val] = s.loc[mask_val].to_numpy() + noise

            print(
                f"- value{val}Proportion{ratio*100:.2f}%, number of perturbation samples:{n_val}"
            )

        df_clean[col] = s

    # 4) ()
    print("\\n[DataClean] Step 4: Shuffle the order of samples ...")
    df_clean = df_clean.sample(frac=1.0, random_state=random_state).reset_index(
        drop=True
    )
    print(f"[DataClean] Number of samples after scrambling:{len(df_clean)}")

    # 4.1 ,
    if "ADCODE99" in df_clean.columns:
        df_clean["ADCODE99"] = _normalize_integer_column(df_clean["ADCODE99"], "ADCODE99")
    if "Disaster" in df_clean.columns:
        df_clean["Disaster"] = _normalize_integer_column(df_clean["Disaster"], "Disaster")

    # 5) Generate output file name (requires 11 file name + _cleaned)
    csv_name = os.path.basename(df_path)         # Positive_Negative_balanced_25366.csv
    csv_stem, _ = os.path.splitext(csv_name)     # Positive_Negative_balanced_25366

    # File name logic for requirement 11 (consistent with aggregate_features):
    # AllFeatures_<csv_stem>.csv or AllFeatures_<csv_stem>_<ssp>.csv
    if ssp is not None and str(ssp).strip() != "":
        base_filename = f"AllFeatures_{csv_stem}_{ssp}.csv"
    else:
        base_filename = f"AllFeatures_{csv_stem}.csv"

    # Requirement 12 Output: Add _cleaned on this basis
    clean_filename = base_filename.replace(".csv", "_cleaned.csv")

    os.makedirs(output_folder_path, exist_ok=True)
    output_path = os.path.join(output_folder_path, clean_filename)

    df_clean.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\\n[DataClean] Clean + scramble completed!")
    print("[DataClean] :")
    print("  ", output_path)
    print(f"[DataClean] :{len(df_clean)}")
    print(f"[DataClean] Final number of features (including original columns):{len(df_clean.columns)}")

    return df_clean
