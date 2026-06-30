"""Data aggregation and analytical processing utilities for the retail sales analysis project."""

from __future__ import annotations
import pandas as pd 

def get_category_signature_data(df: pd.DataFrame) -> pd.DataFrame:
    """Compute normalized performance metrics per product category for signature analysis.

    Aggregates raw revenue drivers and applies Min-Max normalization across 
    categories to scale performance metrics uniformly between 0 and 1.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned input retail transactions DataFrame.

    Returns
    -------
    pd.DataFrame
        Transposed matrix of normalized features (Index: Metrics, Columns: Categories)
        ready for heatmap visualization.
    """
    metrics = df.groupby('category').agg(
        avg_unit_price = ('unit_price', 'mean'), avg_quantity = ('quantity', 'mean'), 
        order_count = ('order_id', 'count'), total_revenue = ('total_price', 'sum'))

    norm_metrics = (metrics - metrics.min()) / (metrics.max() - metrics.min())
    
    norm_metrics = norm_metrics.rename(columns = {'avg_unit_price' : 'Avg Unit Price', 
            'avg_quantity' : 'Avg Quantity', 'order_count' : 'Order Count',
            'total_revenue' : 'Total Revenue'})
    
    return norm_metrics.T

def get_regional_decomposition_data(df: pd.DataFrame) -> pd.DataFrame:
    """Compute customer counts, transaction volumes, and average basket sizes per region.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned input retail transactions DataFrame.

    Returns
    -------
    pd.DataFrame
        Aggregated regional metrics with engineered orders per customer ratios.
    """
    regional_data = df.groupby('region', as_index = False).agg(customer_count = ('customer_id', 'nunique'),
                        order_count = ('order_id', 'count'), avg_order_value = ('total_price', 'mean'))
    regional_data['orders_per_customer'] = regional_data['order_count'] / regional_data['customer_count']
    return regional_data.drop(columns = ['order_count'])


def get_discount_vs_rating_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Extract baseline and segmented customer ratings across discount tiers.

    Aggregates average customer feedback ratings against structured discount percentages
    for the entire store, the Electronics category, and the West operating region.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned input retail transactions DataFrame.

    Returns
    -------
    dict[str, pd.DataFrame]
        A dictionary containing 'Overall', 'Electronics', and 'West' dataframes.
    """
    discount_vs_rating_data = {}
    discount_vs_rating_data['overall'] = df.groupby('discount_pct', as_index = False).agg(
        avg_customer_rating = ('customer_rating', 'mean')).sort_values('discount_pct')
    
    segments = {'category' : 'Electronics', 'region' : 'West'}
    for feature, val in segments.items():
        discount_vs_rating_data[val] = df[df[feature] == val].groupby('discount_pct', as_index = False).agg(
            avg_customer_rating = ('customer_rating', 'mean')).sort_values('discount_pct')
    
    return discount_vs_rating_data

def get_product_concentration_data(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Calculate overall revenue concentration and the share of the top 5 flagship products.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned input retail transactions DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, float]
        - A sorted DataFrame of all products, their revenue, and category.
        - The percentage share contribution of the Top 5 products to TOTAL company revenue.
    """
    product_revenue = df.groupby(['product', 'category'], as_index = False).agg(
        total_revenue = ('total_price', 'sum')).sort_values('total_revenue', ascending = False).reset_index(drop = True)
    total_revenue = df['total_price'].sum()
    product_revenue['share_pct'] = product_revenue['total_revenue'] * 100 / total_revenue
    top_5_product_share = product_revenue.iloc[:5]['share_pct'].sum()
    return product_revenue, top_5_product_share

def get_customer_segmentation_data(df: pd.DataFrame) -> pd.DataFrame:
    """Classify the customer base into tiered spending segments using quantiles.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned input retail transactions DataFrame.

    Returns
    -------
    pd.DataFrame
        Aggregated summary detailing total revenue value and customer counts per segment.
    """
    customer_data = df.groupby('customer_id', as_index = False).agg(total_spend = ('total_price', 'sum'))

    customer_data['customer_class'] = pd.qcut(customer_data['total_spend'], q = [0.0, 0.5, 0.9, 1.0], 
                        labels = ['Standard', 'Regular', 'VIP'])

    customer_class_data = customer_data.groupby('customer_class', observed = True, as_index = False).agg(
        total_revenue = ('total_spend', 'sum'), customer_count = ('customer_id', 'count'))
    
    return customer_class_data

def get_june_rebound_data(df: pd.DataFrame) -> pd.DataFrame:
    """Isolate and aggregate category-level revenue performance for the Q2 2024 rebound period.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned input retail transactions DataFrame.

    Returns
    -------
    pd.DataFrame
        Aggregated metrics (revenue and volume counts) for April, May, and June 2024.
    """
    rebound_df = df.copy()
    rebound_df['order_month'] = rebound_df['order_date'].dt.to_period('M').dt.to_timestamp()
    target_months = pd.to_datetime(["2024-04-01", "2024-05-01", "2024-06-01"])
    filtered_df = rebound_df[rebound_df['order_month'].isin(target_months)]

    aggregated_data = filtered_df.groupby(['order_month', 'category'], as_index = False).agg(
        order_count = ('order_id', 'count'), total_revenue = ('total_price', 'sum'))

    aggregated_data['month_label'] = aggregated_data['order_month'].dt.strftime("%b %Y")
    return aggregated_data