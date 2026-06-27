"""Data cleaning utilities for the retail sales analysis project."""

from __future__ import annotations

from typing import Sequence

from pathlib import Path

import pandas as pd

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
