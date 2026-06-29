"""
data_loader.py — list of helper functions for data loading and preprocessing

Panama dataset: Download dataset once with
    kaggle datasets download -d saurabhshahane/electricity-load-forecasting \
        -p data/ --unzip
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from distutils.util import strtobool
from entsoe import EntsoePandasClient

from config import DATA_DIR, CONTEXT_LENGTH, PREDICTION_LENGTH, N_DAYS, ID_COLUMN, TIMESTAMP_COLUMN, TARGET_COLUMN

# ── 0. Dataset specific configurations ─────────────────────────────────────────
# ── 0.1. Panama dataset ────────────────────────────────────────────────────────
PAN_TIMESTAMP_COL  = "datetime"
PAN_TARGET_COL     = "nat_demand"
PAN_COVARIATE_COLS = [
    "T2M_toc", "QV2M_toc", "TQL_toc", "W2M_toc", 
    "T2M_san", "QV2M_san", "TQL_san", "W2M_san", 
    "T2M_dav", "QV2M_dav", "TQL_dav", "W2M_dav", 
    "Holiday_ID", "holiday", "school"
]   
PAN_START_DATE = "2015-01-04 00:00:00"

# ── 0.2. Australia dataset ─────────────────────────────────────────────────────
AUS_START_DATE = "2010-01-04 00:00:00"

# ── 0.3. Latvia dataset ──────────────────────────────────────────────────────── 
LAT_COUNTRY_CODE = 'LV'  # BZN|LV or Area Code for Latvia
LAT_START_DATE = "2021-01-01"
LAT_END_DATE = "2026-01-01"
LAT_TZ = 'Europe/Riga'

# ── 1. Dataset loading ─────────────────────────────────────────────────────────
# ── 1.1. Panama dataset ────────────────────────────────────────────────────────
def pan_load(path: Path = DATA_DIR / "raw" / "continuous_dataset.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    pan_context_df, pan_horizon_true
    """
    df = pd.read_csv(path, parse_dates=[PAN_TIMESTAMP_COL])
    df = df.sort_values(PAN_TIMESTAMP_COL).reset_index(drop=True)

    # Forward-fill covariate NaNs
    df = df.ffill()

    return df

# ── 1.2. Australia dataset ─────────────────────────────────────────────────────
# Converts the contents in a .tsf file into a dataframe and returns it along with other meta-data of the dataset: frequency, horizon, whether the dataset contains missing values and whether the series have equal lengths
#
# Parameters
# full_file_path_and_name - complete .tsf file path
# replace_missing_vals_with - a term to indicate the missing values in series in the returning dataframe
# value_column_name - Any name that is preferred to have as the name of the column containing series values in the returning dataframe
def aus_convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )

# ── 1.3. Latvia dataset ────────────────────────────────────────────────────────
def lat_load(api_key: str = "cf169f50-54bf-41f4-b6ea-561887e05659") -> pd.DataFrame:
    """
    Downloads and preprocesses the Actual Total Load data for Latvia using the ENTSO-E API.

    Parameters
    ----------
    api_key : str
        The authorized ENTSO-E RESTful API token.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the sorted, timezone-localized Latvian electricity load.
    """
    # Initialize the ENTSO-E Pandas client
    client = EntsoePandasClient(api_key=api_key)

    # Localize to Europe/Riga to respect daylight savings and local system clock
    start = pd.Timestamp(LAT_START_DATE, tz=LAT_TZ)
    end = pd.Timestamp(LAT_END_DATE, tz=LAT_TZ)
    
    try:
        # Fetch the historical actual total grid load
        df = client.query_load(LAT_COUNTRY_CODE, start=start, end=end)
        
        # Convert the Series or raw DataFrame output into a clean standard format
        if isinstance(df, pd.Series):
            df = df.to_frame(name=TARGET_COLUMN)
        elif 'Actual Load' in df.columns:
            df = df.rename(columns={'Actual Load': TARGET_COLUMN})
        else:
            raise KeyError(f"Could not identify load column. Available columns: {list(df.columns)}")
            
        # Reset index to turn the DatetimeIndex into a timestamp column
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: TIMESTAMP_COLUMN})
        
        # Ensure standard sorting
        df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
        
        # Check for any missing values in the target column before filling
        missing_count = df[TARGET_COLUMN].isna().sum()
        if missing_count > 0:
            print(f"[Warning] Found {missing_count} missing entries in '{TARGET_COLUMN}'.")
        else:
            print(f"No missing entries found in '{TARGET_COLUMN}'. Pipeline is clean.")
            return df

    except Exception as e:
        raise RuntimeError(f"Failed to fetch ENTSO-E data for Latvia: {e}")

# ── 2. Dataset preprocessing ───────────────────────────────────────────────────
# ── 2.1. Panama dataset ────────────────────────────────────────────────────────
def pan_preprocessing(
    pan_loaded_data: pd.DataFrame
) -> None:
    """
    Extracts the continuous national demand sequence from the Panama dataset 
    and generates rolling windows across all available complete days for 2016.
    """
    # 1. Initialize lists to accumulate data for all rolling windows
    all_contexts = []
    all_horizons = []

    # 2. Extract relevant columns and set datetime index
    # Isolate time series using original hourly timestamps
    hourly_series = pd.Series(
        pan_loaded_data[PAN_TARGET_COL].values, 
        index=pan_loaded_data[PAN_TIMESTAMP_COL]
    )

    # 3. Implement the rolling window over 365 x 5 days
    start_anchor = pd.Timestamp(PAN_START_DATE)

    # 4. Implement the rolling window step-by-step
    for day in range(N_DAYS):
        # Calculate sliding windows stepping by 24 hours at each iteration
        window_start = start_anchor + pd.Timedelta(hours=day * 24)
        context_end = window_start + pd.Timedelta(hours=CONTEXT_LENGTH)
        horizon_end = context_end + pd.Timedelta(hours=PREDICTION_LENGTH)

        # Extract context (e.g., 168h)
        context_part = hourly_series.loc[window_start : context_end - pd.Timedelta(hours=1)].reset_index()
        context_part.columns = ["timestamp", "target"]
        
        # Create unique ID per rolling window to keep them separate in Chronos-2
        context_part["id"] = f"PAN_day_{day}"

        # Extract ground truth horizon (e.g., 24h)
        horizon_part = hourly_series.loc[context_end : horizon_end - pd.Timedelta(hours=1)].values

        all_contexts.append(context_part)
        all_horizons.append(horizon_part)

    # 5. Combine all series context into a single long-format DataFrame
    pan_context_df = pd.concat(all_contexts, ignore_index=True)

    # 6. Stack horizons into a 2D array of shape (max_days, PREDICTION_LENGTH)
    pan_horizon_true = np.array(all_horizons)

    # 7. Check for missing or infinite values
    assert not pan_context_df["target"].isna().any(), "Panama context contains NaN values!"

    # 8. Save intermediate outputs
    pan_context_df.to_csv(DATA_DIR / "processed" / "pan_context_df.csv", index=False)
    np.save(DATA_DIR / "processed" / "pan_horizon_true.npy", pan_horizon_true)

    # 9. Final confirmation message
    print("\n[SUCCESS] Preprocessing complete! The dataset is ready to be fed to Chronos-2.")

# ── 2.2. Australia dataset ─────────────────────────────────────────────────────
def aus_preprocessing(
    aus_loaded_data: pd.DataFrame
) -> None:
    """
    Extracts the South Australia (SA) series from the pre-loaded Australian dataset,
    downsamples from 30-min to hourly intervals, and generates rolling windows for 2014.
    """
    # 1. Initialize lists to accumulate data for all rolling windows
    all_contexts = []
    all_horizons = []

    # 2. Extract the South Australia (SA) series in the loaded dataframe
    series_row = aus_loaded_data.iloc[3]
    state_id = series_row["state"]
    start_date = series_row["start_timestamp"]
    series_values = pd.Series(series_row["series_value"])

    # Reconstruct the full hourly DatetimeIndex and downsample to hourly
    full_index = pd.date_range(start=start_date, periods=len(series_values), freq="30min")
    series_values.index = full_index
    hourly_series = series_values.resample("h").mean()

    # 3. Implement the rolling window over 365 x 5 days
    start_anchor = pd.Timestamp(AUS_START_DATE)

    for day in range(N_DAYS):
        # Calculate sliding windows stepping by 24 hours at each iteration
        window_start = start_anchor + pd.Timedelta(hours=day * 24)
        context_end = window_start + pd.Timedelta(hours=CONTEXT_LENGTH)
        horizon_end = context_end + pd.Timedelta(hours=PREDICTION_LENGTH)

        # Extract context (168h)
        context_part = hourly_series.loc[window_start : context_end - pd.Timedelta(hours=1)].reset_index()
        context_part.columns = [TIMESTAMP_COLUMN, TARGET_COLUMN]

        # Create unique ID per rolling window (e.g., "AUS_SA_day_0") to keep them separate in Chronos-2
        context_part[ID_COLUMN] = f"AUS_{state_id}_day_{day}"

        # Extract ground truth horizon (24h)
        horizon_part = hourly_series.loc[context_end : horizon_end - pd.Timedelta(hours=1)].values

        all_contexts.append(context_part)
        all_horizons.append(horizon_part)
    
    # 3. Combine all series context into a single long-format DataFrame
    aus_context_df = pd.concat(all_contexts, ignore_index=True)

    # 4. Stack horizons into a 2D array of shape (N_DAYS, PREDICTION_LENGTH)
    aus_horizon_true = np.array(all_horizons)

    # 5. Check missing or infinite values
    assert not aus_context_df[TARGET_COLUMN].isna().any(), "Context contains NaN values!"

    # 6. Save intermediate outputs
    aus_context_df.to_csv(DATA_DIR / "processed" / "aus_context_df.csv", index=False)
    np.save(DATA_DIR / "processed" / "aus_horizon_true.npy", aus_horizon_true)

    # 7. Display previews of the generated intermediate outputs
    # print("--- Context DataFrame Preview (First & Last 5 Rows) ---")
    # display(aus_context_df)

    # print("\n--- Ground Truth (horizon_true) Array Preview ---")
    # print(f"Array Shape: {aus_horizon_true.shape}")
    # print(aus_horizon_true)

    # 8. Final confirmation message
    print("\n[SUCCESS] Preprocessing complete! The dataset is ready to be fed to Chronos-2.")

# ── 2.3. Latvia dataset ────────────────────────────────────────────────────────
def lat_preprocessing(
    lat_loaded_data: pd.DataFrame
) -> None:
    """
    Extracts the continuous total grid load sequence from the Latvian dataset 
    and generates rolling windows across all available complete days.
    """
    # 1. Initialize lists to accumulate data for all rolling windows
    all_contexts = []
    all_horizons = []

    # 2. Convert to naive local timestamps to standardize the clock
    df = lat_loaded_data.copy()
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN]).dt.tz_localize(None)

    # 3. Set index to handle physical data gaps and duplicates
    df = df.set_index(TIMESTAMP_COLUMN)

    # 4. Fix Autumn DST (Duplicate hours): Group by exact time and average the duplicate load values
    df = df.groupby(df.index).mean()

    # 5. Fix Spring DST (Missing hours): Force a strict hourly grid and interpolate the missing hour
    df = df.resample('1h').interpolate(method='linear')

    # The dataset is now a physically perfect 24-hour-per-day grid.
    hourly_series = df[TARGET_COLUMN]

    # 6. Implement the rolling window over N_DAYS matching the target year anchor
    start_anchor = pd.Timestamp(LAT_START_DATE)

    # 7. Implement the rolling window step-by-step
    for day in range(N_DAYS):
        # Calculate sliding windows stepping by 24 hours at each iteration
        window_start = start_anchor + pd.Timedelta(hours=day * 24)
        context_end = window_start + pd.Timedelta(hours=CONTEXT_LENGTH)
        horizon_end = context_end + pd.Timedelta(hours=PREDICTION_LENGTH)

        # Extract context (e.g., 168h)
        context_part = hourly_series.loc[window_start : context_end - pd.Timedelta(hours=1)].reset_index()
        context_part.columns = [TIMESTAMP_COLUMN, TARGET_COLUMN]

        # Create unique ID per rolling window to keep them separate in Chronos-2
        context_part[ID_COLUMN] = f"LAT_day_{day}"

        # Extract ground truth horizon (e.g., 24h)
        horizon_part = hourly_series.loc[context_end : horizon_end - pd.Timedelta(hours=1)].values

        # Safety Check: If a window is clipped due to data boundaries, raise a clear error
        if len(horizon_part) != PREDICTION_LENGTH:
            raise ValueError(
                f"Day {day} generated a horizon of length {len(horizon_part)}, "
                f"expected {PREDICTION_LENGTH}. Window: {context_end} to {horizon_end}"
            )
        
        all_contexts.append(context_part)
        all_horizons.append(horizon_part)

    # 8. Combine all series context into a single long-format DataFrame
    lat_context_df = pd.concat(all_contexts, ignore_index=True)

    # 9. Stack horizons into a 2D array of shape (N_DAYS, PREDICTION_LENGTH)
    lat_horizon_true = np.array(all_horizons)

    # 10. Check for missing or infinite values
    assert not lat_context_df[TARGET_COLUMN].isna().any(), "Latvia context contains NaN values!"

    # 11. Save intermediate outputs
    lat_context_df.to_csv(DATA_DIR / "processed" / "lat_context_df.csv", index=False)
    np.save(DATA_DIR / "processed" / "lat_horizon_true.npy", lat_horizon_true)

    # 12. Final confirmation message
    print("\n[SUCCESS] Preprocessing complete! The dataset is ready to be fed to Chronos-2.")