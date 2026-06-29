"""Data visualization and plotting utilities for the retail sales analysis project."""

from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.dates as mdates
import seaborn as sns


def plot_distribution_with_outliers(
    df: pd.DataFrame,
    column: str,
    discrete: bool = True,
    kde: bool = True,
    figsize: tuple[float, float] = (10, 6),
) -> None:
    """Plot a boxplot above a histogram on a shared x-axis for outlier review.

    The boxplot exposes the IQR fences and fliers while the aligned histogram
    shows the underlying shape (skew, modes), making it easy to decide between
    mean and median imputation for a numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    column : str
        Numeric column to visualize (e.g., 'customer_age', 'customer_rating').
    discrete : bool, default=True
        Treat values as discrete integers in the histogram (one bar per value).
    kde : bool, default=True
        Overlay a kernel density estimate on the histogram.
    figsize : tuple of float, default=(10, 6)
        Figure size in inches.

    Returns
    -------
    None
        Renders the figure via plt.show().

    """
    
    fig, (box_ax, hist_ax) = plt.subplots(nrows = 2, ncols = 1,sharex = True, gridspec_kw = {"height_ratios": [0.2, 0.8]}, dpi = 150, figsize = figsize)

    sns.boxplot(data = df, x = column, ax = box_ax, width = 0.5, fliersize = 5)
    sns.histplot(data = df, x = column, ax = hist_ax, discrete = discrete, kde = kde)
    
    column = column.replace('_', ' ').title()
    hist_ax.set_xlabel(column, fontsize = 12, labelpad = 10)
    hist_ax.set_ylabel('Frequency / Density', fontsize = 12, labelpad = 10)

    plt.suptitle(f'{column} Distribution & Outlier Analysis', fontsize = 14, fontweight = "bold")
    plt.tight_layout()
    plt.show()

def plot_max_deviations(
    max_deviations: dict[str, float],
    threshold: float,
    figsize: tuple[int, int] = (8, 5)
) -> None:
    """
    Visualize maximum deviation percentages with threshold line.
    
    Parameters
    ----------
    features_max_deviation_pct : dict[str, float]
        Dictionary mapping feature names to deviation percentages.
    threshold : float
        Deviation threshold (%). Features above this proceed to Stage 2.
    figsize : tuple[int, int], default=(8, 5)
        Figure size (width, height) in inches.
    
    Returns
    -------
    None
        Displays the plot via plt.show().
        
    """
    max_deviations = pd.DataFrame({'feature' : max_deviations.keys(),
                        'max_deviation_pct' : max_deviations.values()})
    palette = max_deviations['max_deviation_pct'].apply(
        lambda pct: "#e74c3c" if pct >= threshold else "#3498db").tolist()
    
    fig, ax = plt.subplots(dpi = 200, figsize = figsize)
    sns.barplot(data = max_deviations, x = 'feature', y = 'max_deviation_pct', ax = ax,
                hue = 'feature', palette = palette, legend = False, edgecolor = 'black', linewidth = 0.7 )
    for i in range(len(max_deviations)):
        pct = max_deviations.loc[i, "max_deviation_pct"]
        ax.text(x = i, y = pct + 0.3, s = f"{pct:.1f}%", fontsize = 10, ha = 'center', va = 'bottom')
    ax.axhline(threshold, color = 'red', linestyle = '--', linewidth = 1.5, label = f"{threshold:.0f}% Threshold")
    
    ax.set_xlabel('Grouping Feature', fontsize = 12, labelpad = 10)
    ax.set_ylabel('Max Deviation (%)', fontsize = 12, labelpad = 10)
    ax.set_title('Stage 1: Maximum Group Deviation from Global Median', fontweight = 'bold', fontsize = 14, pad = 15)
    ax.set_ylim(0, max(max_deviations['max_deviation_pct'].max(), threshold) + 2)
    ax.legend(loc = 'upper right')

    plt.tight_layout()
    plt.show()

def style_axis(
    ax: matplotlib.axes.Axes,
    x_label: str | None,
    y_label: str | None,
    title: str,
    label_fontsize: int = 12,
    title_fontsize: int = 14,
    title_fontweight: str = 'bold',
) -> None:
    """
    Apply consistent axis labels and title to a subplot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The subplot to style (e.g., ax[3][1]).
    x_label : str or None
        Label for the x-axis. If None, no label is set.
    y_label : str or None
        Label for the y-axis. If None, no label is set.
    title : str
        Title for the subplot.
    label_fontsize : int, default=12
        Font size for both x and y axis labels.
    title_fontsize : int, default=14
        Font size for the title.
    title_fontweight : str, default='bold'
        Font weight for the title.
    """
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize = label_fontsize)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize = label_fontsize)
    ax.set_title(title, fontsize = title_fontsize, fontweight = title_fontweight)

def plot_revenue_by_category(
    df: pd.DataFrame,
    ax: matplotlib.axes.Axes,
) -> None:
    """
    Bar chart showing total revenue for each product category.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned retail sales dataframe with 'category' and 'total_price' columns.
    ax : matplotlib.axes.Axes
        The subplot axis on which to draw the chart.
    """
    revenue_by_cat = df.groupby('category', as_index = False).agg(total_revenue = ('total_price', 'sum'))
    sns.barplot(data = revenue_by_cat, x = 'category',y = 'total_revenue', ax = ax, hue = 'category',
        legend = False, palette = 'Blues_r')
    style_axis(ax, 'Product Category', 'Total Revenue ($)', 'Revenue by Category')
    ax.ticklabel_format(style = 'plain', axis = 'y')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v/1e6:.1f}M'))

def plot_monthly_revenue_trend(
    df: pd.DataFrame,
    ax: matplotlib.axes.Axes,
) -> None:
    """
    Line chart showing monthly revenue trend from 2023 to 2024.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned retail sales dataframe with 'order_date' and 'total_price' columns.
    ax : matplotlib.axes.Axes
        The subplot axis on which to draw the chart.
    """
    df['order_Y-M'] = df['order_date'].dt.strftime('%Y-%m')
    monthly_revenue = df.groupby('order_Y-M', as_index = False, observed = True).agg(total_revenue = ('total_price', 'sum'))
    sns.lineplot(data = monthly_revenue, x = 'order_Y-M', y = 'total_revenue', ax = ax,
                marker = "o", color = "#1f77b4", linewidth = 2.5 )
    style_axis(ax, 'Timeline', 'Total Revenue ($)', 'Monthly Revenue Trend (2023-2024)')
    ax.tick_params(axis = 'x', rotation = 90)

def plot_payment_method_share(
    df: pd.DataFrame,
    ax: matplotlib.axes.Axes,
) -> None:
    """
    Pie chart showing the share of orders by payment method.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned retail sales dataframe with 'payment_method' column.
    ax : matplotlib.axes.Axes
        The subplot axis on which to draw the chart.
    """
    pm_order_share = df['payment_method'].value_counts()
    ax.pie(pm_order_share.values, labels = pm_order_share.index, autopct = "%1.1f%%", 
                startangle = 90, colors = sns.color_palette("Pastel1"),
                wedgeprops = {'edgecolor' : 'black', 'linewidth' : 0.5})
    style_axis(ax, None, None, 'Order Share by Payment Method')

def plot_top_products_by_quantity(
    df: pd.DataFrame,
    ax: matplotlib.axes.Axes,
) -> None:
    """
    Horizontal bar chart of top 10 products by total quantity sold.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned retail sales dataframe with 'product' and 'quantity' columns.
    ax : matplotlib.axes.Axes
        The subplot axis on which to draw the chart.
    """
    top_products = df.groupby('product', as_index = False).agg(total_quantity = ('quantity', 'sum')).sort_values(
        'total_quantity', ascending = False).reset_index(drop = True).iloc[:10]
    sns.barplot(data = top_products, x = 'total_quantity', y = 'product', hue = 'product', 
                legend = False, ax = ax, palette = "viridis")
    style_axis(ax, 'Quantity', '', 'Top 10 Products by Quantity')

def plot_customer_age_distribution(
    df: pd.DataFrame,
    ax: matplotlib.axes.Axes,
) -> None:
    """
    Histogram showing the distribution of customer ages with KDE overlay.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned retail sales dataframe with 'customer_age' column.
    ax : matplotlib.axes.Axes
        The subplot axis on which to draw the chart.
    """
    sns.histplot(data = df, x = 'customer_age', discrete = True, ax = ax, kde = True, color = "#2ca02c")
    style_axis(ax, 'Customer Age', 'Frequency / Density', 'Customer Age Distribution')

def plot_transaction_value_by_region(
    df: pd.DataFrame,
    ax: matplotlib.axes.Axes,
) -> None:
    """
    Boxplot of transaction values grouped by region, highlighting outliers.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned retail sales dataframe with 'region' and 'total_price' columns.
    ax : matplotlib.axes.Axes
        The subplot axis on which to draw the chart.
    """
    sns.boxplot(data = df, x = 'region', y = 'total_price', ax = ax, flierprops = {"markerfacecolor": "red", "marker": "D", "markersize": 5})
    style_axis(ax, 'Geographic Region', 'Order Transaction Value ($)', 'Transaction Value by Region')

def plot_rating_vs_discount(
    df: pd.DataFrame,
    ax: matplotlib.axes.Axes,
) -> None:
    """
    Scatter plot (jittered stripplot) of discount percentage vs customer rating.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned retail sales dataframe with 'discount_pct' and 'customer_rating' columns.
    ax : matplotlib.axes.Axes
        The subplot axis on which to draw the chart.
    """
    sns.stripplot(data = df, x = 'discount_pct', y = 'customer_rating', ax = ax, 
                color = '#e377c2', alpha = 0.15, jitter = 0.25, size = 3)
    style_axis(ax, 'Discount Percentage (%)', 'Customer Rating (Stars)', 'Rating vs. Discount')

def plot_revenue_by_region_and_category(
    df: pd.DataFrame,
    ax: matplotlib.axes.Axes,
) -> None:
    """
    Grouped bar chart showing revenue by region, segmented by category.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned retail sales dataframe with 'region', 'category', and 'total_price' columns.
    ax : matplotlib.axes.Axes
        The subplot axis on which to draw the chart.
    """
    regional_cat_revenue = df.groupby(['region', 'category'], observed = True, as_index = False).agg(total_revenue = ('total_price', 'sum'))
    sns.barplot(data = regional_cat_revenue, x = 'region', y = 'total_revenue', hue = 'category', 
                ax = ax, palette = "tab10")
    style_axis(ax, 'Geographic Region', 'Total Revenue ($)', 'Revenue by Region and Category')
    ax.legend(title = 'Categories', bbox_to_anchor = (1.05, 1), loc = "upper left")

def plot_monthly_revenue_with_rolling(
    df: pd.DataFrame,
    monthly_ax: matplotlib.axes.Axes,
) -> None:
    """
    Line chart showing monthly revenue trend with a 30-day rolling average overlay.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned retail sales dataframe with 'order_date' and 'total_price' columns.
    ax : matplotlib.axes.Axes
        The primary subplot axis on which to draw the monthly chart.
    """
    df['order_Y-M'] = df['order_date'].dt.to_period('M').dt.to_timestamp()
    monthly_revenue = df.groupby('order_Y-M', as_index = False).agg(total_revenue = ('total_price', 'sum'))
    sns.lineplot(data = monthly_revenue, x = 'order_Y-M', y = 'total_revenue', ax = monthly_ax,
                zorder = 3, color = "#1f77b4", marker = "o", linewidth = 2.5, label = "Monthly Revenue")

    rolling_ax = monthly_ax.twinx()
    rolling_ax.grid(False)

    daily_revenue = df.groupby('order_date', as_index = False).agg(total_revenue = ('total_price', 'sum'))
    daily_revenue['rolling_30d'] = daily_revenue['total_revenue'].rolling(window = 30, min_periods = 1).mean()
    sns.lineplot(data = daily_revenue, x = 'order_date', y = 'rolling_30d', ax = rolling_ax, 
                zorder = 2, color = '#bcbd22', label = '30-Day Rolling Avg')

    monthly_ax.set_xlabel('Timeline', fontsize = 12)
    monthly_ax.tick_params(axis = 'x', rotation = 90)
    monthly_ax.set_ylabel('Total Monthly Revenue ($)', fontsize = 12)
    monthly_ax.set_title("Monthly Revenue Trend with 30-Day Daily Rolling Average (2023-2024)",
                        fontsize = 14, fontweight = 'bold', pad = 15)
    monthly_ax.set_xticks(monthly_revenue['order_Y-M'])
    monthly_ax.margins(x = 0.05)
    monthly_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    rolling_ax.set_ylabel('Daily Rolling Average ($)', fontsize = 12)

    handles_1, labels_1 = monthly_ax.get_legend_handles_labels()
    handles_2, labels_2 = rolling_ax.get_legend_handles_labels()

    if monthly_ax.get_legend() is not None:
        monthly_ax.get_legend().remove()
    if rolling_ax.get_legend() is not None:
        rolling_ax.get_legend().remove()

    monthly_ax.legend(handles_1 + handles_2, labels_1 + labels_2, loc = 'upper left', 
                    bbox_to_anchor = (1.05, 1))