"""Data cleaning utilities for the retail sales analysis project."""

from __future__ import annotations
from typing import Sequence
from pathlib import Path
import pandas as pd
import numpy as np

def load_data(filepath: str = "../data/raw/retail_sales.csv") -> pd.DataFrame:
    """Load the raw retail sales CSV.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        The raw, unmodified DataFrame.
    """
    
    return pd.read_csv(filepath, low_memory = False)

def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove fully duplicated rows.

    Performed early so downstream statistics are computed on
    de-duplicated data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, int]
        The de-duplicated DataFrame and the count of removed rows.
    """
    
    num_dups = df.duplicated(keep = 'first').sum()
    df.drop_duplicates(keep = 'first', inplace = True)
    return (df, num_dups)

def detect_missing_values(df: pd.DataFrame) -> pd.Series:
    """Report the number of missing values per column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.Series
        Columns with at least one missing value (index = column name,
        value = count), sorted descending. Empty if no missing values.
    """

    cols_missing_cnt = df.isnull().sum()
    return cols_missing_cnt[cols_missing_cnt > 0].sort_values(ascending = False)    

def print_missing_value_intersection(df: pd.DataFrame) -> None:
    """
    Analyzes and prints the intersection of missing values across 
    customer_age, customer_rating, and payment_method columns.
    
    Args:
        df: The pandas DataFrame containing the retail data.
    """
    ca_nans = df["customer_age"].isnull()
    cr_nans = df["customer_rating"].isnull()
    pm_nans = df["payment_method"].isnull()

    ca_cr = (ca_nans & cr_nans).sum()
    ca_pm = (ca_nans & pm_nans).sum()
    cr_pm = (cr_nans & pm_nans).sum()
    all_three = (ca_nans & cr_nans & pm_nans).sum()

    print("Missing value intersection counts:")
    print(f"customer_age ∩ customer_rating: {ca_cr}")
    print(f"customer_age ∩ payment_method: {ca_pm}")
    print(f"customer_rating ∩ payment_method: {cr_pm}")
    print(f"All three columns: {all_three}")

    total_missing = (ca_nans | cr_nans | pm_nans).sum()
    pct_missing = (total_missing / len(df)) * 100
    print(f"Using dropna() would remove: {total_missing} rows ({pct_missing:.2f}%)")

def print_missing_report(
    df: pd.DataFrame,
    column: str,
    missing_count: int,
) -> None:
    """
    Print a formatted missing-value summary for a single column.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset being analyzed.
    column : str
        Name of the column to report on.
    missing_count : int
        Pre-computed count of missing values for the column (from missing_report series).

    Returns
    -------
    None
        Prints formatted output to stdout.
    """
    
    missing_pct = missing_count * 100 / len(df)
    print(f"'{column}' has {missing_count} missing values ({missing_pct:.2f}% of total rows).")

def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    """Detect outliers using the Interquartile Range (IQR) method.

    Values below Q1 - 1.5×IQR or above Q3 + 1.5×IQR are flagged
    as outliers. This is a robust method that is not affected by
    the outliers themselves.

    Parameters
    ----------
    series : pd.Series
        Numeric series to check for outliers.

    Returns
    -------
    pd.Series
        Boolean series where True indicates an outlier.
    """
    
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_fence = Q1 - IQR * 1.5 
    upper_fence = Q3 + IQR * 1.5 
    return (series < lower_fence) | (series > upper_fence)

def compute_max_deviations(
    df: pd.DataFrame,
    target_column: str,
    grouping_features: list[str],
) -> dict[str, float]:
    """
    Compute maximum percentage deviation of group medians from global median.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_column : str
        Column to analyze (e.g., 'customer_age', 'customer_rating').
    grouping_features : list[str]
        List of features to test for group-based imputation.
 
    Returns
    -------
    dict[str, float]
        Dictionary mapping feature name to maximum deviation percentage.
    """
    
    ind = df[~df[target_column].isnull()].index
    global_median = df[target_column].median()
    max_deviations = {}
    for feature in grouping_features:
        max_deviations[feature] = ((((df.loc[ind].groupby(feature, observed = True)[target_column].median()) - global_median).abs()) * 100 / global_median).max()
    return max_deviations

def holdout_imputation_test(
    df: pd.DataFrame,
    target_column: str,
    grouping_features: list[str],
    threshold: float,
    test_size: float = 0.30,
    statistic: str = "median",
    metric: str = "mae",
    random_state: int = 42,
) -> pd.DataFrame:
    """Stage 2 holdout test comparing global vs. group-based imputation lift.

    Masks a random holdout of known target values, recomputes imputation
    statistics from the remaining training portion only, and measures the
    predictive lift of each candidate grouping feature against the global
    baseline. Numeric targets are scored with Mean Absolute Error (MAE) and
    categorical targets with classification accuracy. Only features whose
    improvement meets the threshold are retained, justifying the added
    pipeline complexity of group-based structures.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the target column and grouping features.
    target_column : str
        Column to impute (e.g., 'customer_age', 'customer_rating', 'payment_method').
    grouping_features : list of str
        Candidate features to test for group-based imputation (e.g., ['product']).
    threshold : float
        Minimum required improvement over the global baseline for a feature to pass:
        - For MAE: global_MAE - group_MAE must be >= threshold (error reduction).
        - For accuracy: group_accuracy - global_accuracy must be >= threshold.
    test_size : float, default=0.30
        Proportion of known target values to mask as the holdout set.
    statistic : {"median", "mode"}, default="median"
        Summary statistic used for imputation:
        - "median": For numeric targets (e.g., customer_age, customer_rating).
        - "mode": For categorical targets (e.g., payment_method).
    metric : {"mae", "accuracy"}, default="mae"
        Evaluation metric:
        - "mae": Mean Absolute Error for numeric targets (lower is better).
        - "accuracy": Proportion of correct predictions for categorical targets.
    random_state : int, default=42
        Random seed for the holdout split, for reproducibility.

    Returns
    -------
    pd.DataFrame
        A summary DataFrame containing the evaluation results for each tested 
        grouping feature. Rows are sorted best-first (ascending order for MAE 
        or descending order for accuracy).

    """

    data_idx = df[~df[target_column].isnull()].index
    test_idx = df.loc[data_idx].sample(frac = test_size, random_state = random_state).index
    features_eval = {}
    for feature in grouping_features + [target_column]:
        data = df.loc[data_idx].copy()
        data.loc[test_idx, target_column] = np.nan
        if feature == target_column:
            data = impute_column(data, target_column, statistic)
        else:
            data = impute_column(data, target_column, statistic, feature)
        
        if metric == "mae":
            features_eval[feature] = (data.loc[test_idx, target_column] - df.loc[test_idx, target_column]).abs().sum() / len(test_idx)
        elif metric == "accuracy":
            features_eval[feature] = (data.loc[test_idx, target_column] == df.loc[test_idx, target_column]).mean()
    
    global_eval = features_eval.pop(target_column)
    print(f'{metric.upper()} Score of {target_column} imputation using global {statistic}: {global_eval}')
    
    features_eval = pd.DataFrame({'feature' : features_eval.keys(), f'{metric}' : features_eval.values()})
    if metric == 'mae':
        features_eval["improvement"] = global_eval - features_eval['mae']
        features_eval["passes"] = features_eval["improvement"] >= threshold
        features_eval = features_eval.sort_values('mae')
    elif metric == 'accuracy':
        features_eval["improvement"] = features_eval['accuracy'] - global_eval
        features_eval["passes"] = features_eval["improvement"] >= threshold
        features_eval = features_eval.sort_values('accuracy', ascending = False)
        
    return features_eval

def mode_based_feature_selection(
    df: pd.DataFrame,
    target_column: str,
    grouping_features: list[str],
    threshold: float = 0.15,
) -> pd.DataFrame:
    """
    Stage 1: Mode-Based Relative Frequency Shift for categorical imputation.

    Evaluates candidate grouping features to determine if they create sub-groups
    with distinct payment class distributions. A feature passes if at least one
    of its groups satisfies either:
      1. Mode Cardinality Flip: Local mode differs from the global mode.
      2. Baseline Frequency Spike: Local mode matches global but its relative
         frequency exceeds the global baseline by >= threshold.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset containing the target column and grouping features.
    target_column : str
        The categorical column to impute (e.g., 'payment_method').
    grouping_features : list[str]
        Candidate features to test as grouping keys.
    threshold : float, default=0.15
        Minimum relative frequency shift required for a same-mode group to pass.

    Returns
    -------
    pd.DataFrame
        A two-column DataFrame with 'feature' names and 'is_selected' boolean
        indicating which features passed Stage 1 and should proceed to holdout.
    """
    
    global_mode = df[target_column].mode()[0]
    non_missing_idx = df[~df[target_column].isnull()].index
    global_mode_rf = (df.loc[non_missing_idx, target_column] == global_mode).mean()
    features_pass = {}

    def group_evaluation(group):
        local_mode = group.mode()[0]
        return (local_mode != global_mode) or (((group == local_mode).mean() - global_mode_rf) >= threshold)
            
    for feature in grouping_features:
        grouped_mode = df.loc[non_missing_idx].groupby(feature, observed = True)[target_column].apply(
            lambda x: group_evaluation(x))
        features_pass[feature] = grouped_mode.any()
            
    return pd.DataFrame({'feature' : features_pass.keys(), 'is_selected' : features_pass.values()})
        
def impute_column(
    df: pd.DataFrame,
    target_column: str,
    statistic: str = "median",
    group_by: str | None = None,
) -> pd.DataFrame:
    """Impute missing values in a target column using global or group-based statistics.

    Supports both numeric (median) and categorical (mode) imputation strategies.
    When group_by is provided, uses per-group statistics with a global fallback
    for groups that have no valid values or when the grouping feature itself is missing.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the target column and optional grouping feature.
    target_column : str
        Name of the column to impute (e.g., 'customer_age', 'customer_rating', 'payment_method').
    statistic : {"median", "mode"}, default="median"
        Summary statistic to use for imputation:
        - "median": For numeric columns (e.g., customer_age, customer_rating).
        - "mode": For categorical columns (e.g., payment_method).
    group_by : str or None, optional
        Column name to condition imputation on (e.g., "customer_id", "product").
        If None, uses global statistic for all missing values.
        If provided, computes per-group statistics and falls back to global
        statistic when:
        - The group has no non-missing values in the target column.
        - The grouping feature itself is missing for a given row.

    Returns
    -------
    pd.DataFrame
        DataFrame with target_column fully populated (no missing values).

    """

    if statistic == "median":
        global_stats = df[target_column].median()
        stats = lambda x: x.median()
    elif statistic == "mode":
        global_stats = df[target_column].mode()[0]
        stats = lambda x: x.mode()[0]
    
    if group_by is None:
        df[target_column] = df[target_column].fillna(global_stats)
    else:
        non_missing_idx = df[~df[target_column].isnull()].index
        missing_idx = df[df[target_column].isnull()].index
        grouped_stats = df.loc[non_missing_idx].groupby(group_by, observed = True)[target_column].apply(stats)
        df.loc[missing_idx, target_column] = df.loc[missing_idx].apply(
            lambda x: global_stats if (pd.isna(x[group_by]) or x[group_by] not in grouped_stats.index) else grouped_stats[x[group_by]], axis = 1)
    return df


def cast_clean_dtypes(
    df: pd.DataFrame,
    int_columns: Sequence[str] = ("customer_age", "customer_rating"),
) -> pd.DataFrame:
    """Cast columns to int64 after imputation.

    Must run only after the target columns have no missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    int_columns : Sequence[str], optional
        Columns to cast to integer.

    Returns
    -------
    pd.DataFrame
        DataFrame with specified columns cast to int64.
    """
    
    for col in int_columns:
        if df[col].isnull().sum() > 0:
            raise ValueError(f"Column '{col}' contain missing values and cannot be cast to int.")
        if (df[col] % 1 != 0).any():
            raise ValueError(f"Column '{col}' contains values with actual fractional parts!")
        df[col] = df[col].astype('int64')
    return df

def parse_dates_and_create_features(
    df: pd.DataFrame,
    date_column: str = "order_date",
) -> pd.DataFrame:
    """Convert order_date to datetime and create calendar features.

    Adds order_month (month name) and order_weekday (day name).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    date_column : str, optional
        Name of the date column to parse.

    Returns
    -------
    pd.DataFrame
        DataFrame with parsed datetime and two new feature columns.
    """
    
    df[date_column] = pd.to_datetime(df[date_column])
    df["order_month"] = df[date_column].dt.month_name()
    df["order_weekday"] = df[date_column].dt.day_name()
    return df


def validate_clean_data(df: pd.DataFrame) -> None:
    """Validate the cleaned DataFrame before export.

    Asserts no missing values, correct dtypes for IDs/age/rating,
    and presence of engineered date features.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned DataFrame.

    Raises
    ------
    AssertionError
        If any validation check fails.
    """
    
    num_missing = df.isnull().sum().sum()
    if num_missing > 0:
        raise ValueError(f"Validation Failed: Dataframe contains {num_missing} missing values!")
        
    if df.duplicated(keep = 'first').any():
        raise ValueError(f"Validation Failed: Found {df.duplicated(keep = 'first').sum()} duplicate rows that should have been removed.")
    
    for col in ["customer_age", "customer_rating"]:
        if df[col].dtype != 'int64':
            raise TypeError(f"Validation Failed: Column '{col}' must be type 'int64'.")
        
    if not pd.api.types.is_datetime64_any_dtype(df["order_date"]):
        raise TypeError(f"Validation Failed: 'order_date' must be a datetime type, got '{df['order_date'].dtype}'.")
    
    for col in ["order_month", "order_weekday"]:
        if col not in df.columns:
            raise KeyError(f"Validation Failed: Column '{col}' not found.")
    
    months = {"January", "February", "March", "April", "May", "June", "July", "August",
        "September", "October", "November", "December"}
    
    weekdays = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}

    invalid_months = set(df["order_month"].unique()) - months
    if invalid_months:
        raise ValueError(f"Validation Failed: 'order_month' contains unexpected names: {invalid_months}")
    
    invalid_weekdays = set(df["order_weekday"].unique()) - weekdays
    if invalid_weekdays:
        raise ValueError(f"Validation Failed: 'order_weekday' contains unexpected names: {invalid_weekdays}")

def save_clean_data(
    df: pd.DataFrame,
    output_path: str = "../data/processed/retail_sales_cleaned.csv",
) -> None:
    """Save the cleaned DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        The validated, cleaned DataFrame.
    output_path : str, optional
        Destination path.
    """
    Path(output_path).parent.mkdir(parents = True, exist_ok = True)
    df.to_csv(output_path, index = False)
