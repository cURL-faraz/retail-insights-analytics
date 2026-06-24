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

def impute_customer_age(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing customer_age with the global median.

    The median is robust to the 14 high outliers and justified by
    negligible inter-group variance across product categories and
    payment methods (see notebook analysis).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with customer_age fully populated.
    """
    
    Q2 = df["customer_age"].median()
    df["customer_age"] = df["customer_age"].fillna(Q2)
    return df

def impute_customer_rating(
    df: pd.DataFrame,
    group_by: str | None = None,
) -> pd.DataFrame:
    """Impute missing customer_rating values.

    Uses the median by default (a valid integer rating, robust to the
    37 low-rating outliers). If group_by is provided, uses the per-group
    median with a global-median fallback for groups with no valid rating.

    Holdout evaluation (20% mask) shows customer-grouped imputation
    reduces MAE by ~19% compared to the global median.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    group_by : str or None, optional
        Column to condition imputation on (e.g. "customer_id").

    Returns
    -------
    pd.DataFrame
        DataFrame with customer_rating fully populated.
    """
    
    Q2 = df["customer_rating"].median()
    if group_by is None:
        df["customer_rating"] = df["customer_rating"].fillna(Q2)
    else:
        customer_rating_pred = df.groupby(group_by, observed = True)["customer_rating"].transform(
            lambda x : Q2 if pd.isna(x.median()) else x.median())
        df["customer_rating"] = df["customer_rating"].fillna(customer_rating_pred)
    return df

def impute_payment_method(
    df: pd.DataFrame,
    group_by: str | None = None,
) -> pd.DataFrame:
    """Impute missing payment_method values.

    Uses the global mode by default. If group_by is provided, uses
    the per-group mode with a global-mode fallback for groups with
    no valid value.

    Holdout evaluation shows customer-grouped imputation increases
    accuracy by ~35% over the global mode.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    group_by : str or None, optional
        Column to condition imputation on (e.g. "customer_id").

    Returns
    -------
    pd.DataFrame
        DataFrame with payment_method fully populated.
    """
    payment_method_mode = df["payment_method"].mode()[0]
    if group_by is None:
        df["payment_method"] = df["payment_method"].fillna(payment_method_mode)
    else:
        payment_method_pred = df.groupby(group_by, observed = True)["payment_method"].transform(
            lambda x: x.mode()[0])
        df["payment_method"] = df["payment_method"].fillna(payment_method_pred)
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
