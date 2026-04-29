# feature_state_io.py
import os
import pickle


def _make_state_filename(df_path, ssp=None):
    """Generate a status file name based on df_path and ssp, and save it in the directory where the current code is located."""
    # .py
    base_dir = os.path.dirname(os.path.abspath(__file__))

    csv_name = os.path.basename(df_path)         # Positive_Negative_balanced_25366.csv
    csv_stem, _ = os.path.splitext(csv_name)     # Positive_Negative_balanced_25366

    if ssp is not None and str(ssp).strip() != "":
        ssp_str = str(ssp).strip().replace(" ", "")
        fname = f"feature_state_{csv_stem}_{ssp_str}.pkl"
    else:
        fname = f"feature_state_{csv_stem}.pkl"

    return os.path.join(base_dir, fname)


def save_feature_state(
    df_path,
    ssp,
    sinkhole_position,
    database_folder_path,
    input_folder_path,
    output_folder_path,
    historical_folder_path,
    future_folder_path,
    future_ssp_folder_path,
    extra_vars=None,
):
    """10:"" pickle . : -------- df_path : str CSV . ssp : str SSP , 'ssp1', 'ssp2', 'ssp3', 'ssp5'. sinkhole_position : pandas.DataFrame 1–10,. database_folder_path, input_folder_path, output_folder_path, historical_folder_path, future_folder_path, future_ssp_folder_path : str . extra_vars : dict, optional ,, state. : ---- state_path : str pickle ."""
    state = {
        "df_path": df_path,
        "ssp": ssp,
        "sinkhole_position": sinkhole_position,
        "database_folder_path": database_folder_path,
        "input_folder_path": input_folder_path,
        "output_folder_path": output_folder_path,
        "historical_folder_path": historical_folder_path,
        "future_folder_path": future_folder_path,
        "future_ssp_folder_path": future_ssp_folder_path,
    }

    if extra_vars is not None:
        # Processing step.
        state.update(extra_vars)

    state_path = _make_state_filename(df_path, ssp)

    with open(state_path, "wb") as f:
        pickle.dump(state, f)

    print("\\n[FeatureState] The global variable state after feature extraction has been saved:")
    print("Path:", state_path)
    print(":", list(state.keys()))

    return state_path


def load_feature_state(df_path, ssp=None):
    """Called in subsequent scripts: read the previously saved status file based on df_path + ssp,
    Returns a dictionary containing all variables.

    Usage examples:
    --------
    state = load_feature_state(df_path, ssp)
    sinkhole_position = state["sinkhole_position"]
    output_folder_path = state["output_folder_path"]
    ..."""
    state_path = _make_state_filename(df_path, ssp)

    if not os.path.exists(state_path):
        raise FileNotFoundError(
            f"[FeatureState] State file not found:{state_path}\n"
            f"Please call save_feature_state to save once after completing requirement 10."
        )

    with open(state_path, "rb") as f:
        state = pickle.load(f)

    print("\\n[FeatureState] The global variable state after feature extraction has been read:")
    print("Path:", state_path)
    print("Variable key:", list(state.keys()))

    return state
