from typing import List, Dict, Any, Callable
from data_structures import CSVRow, DataType
import pandas as pd
import numpy as np
from scipy import stats

def apply_aggregation(data: List[CSVRow], group_by: List[str], agg_func: Dict[str, Callable]) -> List[CSVRow]:
    """
    Apply aggregation functions to grouped data.
    
    :param data: List of CSVRow objects
    :param group_by: List of column names to group by
    :param agg_func: Dictionary of column names and aggregation functions
    :return: List of aggregated CSVRow objects
    """
    df = pd.DataFrame([row._data for row in data])
    grouped = df.groupby(group_by).agg(agg_func).reset_index()
    return [CSVRow(row) for _, row in grouped.iterrows()]

def apply_window_function(data: List[CSVRow], window_func: Callable, column: str, window_size: int) -> List[CSVRow]:
    """
    Apply a window function to a specific column.
    
    :param data: List of CSVRow objects
    :param window_func: Window function to apply (e.g., rolling mean)
    :param column: Column name to apply the function to
    :param window_size: Size of the rolling window
    :return: List of CSVRow objects with the new column added
    """
    df = pd.DataFrame([row._data for row in data])
    df[f'{column}_window'] = df[column].rolling(window=window_size).apply(window_func)
    return [CSVRow(row) for _, row in df.iterrows()]

def detect_anomalies(data: List[CSVRow], column: str, method: str = 'zscore', threshold: float = 3.0) -> List[CSVRow]:
    """
    Detect anomalies in a specific column using various methods.
    
    :param data: List of CSVRow objects
    :param column: Column name to check for anomalies
    :param method: Method to use for anomaly detection ('zscore' or 'iqr')
    :param threshold: Threshold for anomaly detection
    :return: List of CSVRow objects with an additional 'is_anomaly' column
    """
    df = pd.DataFrame([row._data for row in data])
    
    if method == 'zscore':
        z_scores = np.abs(stats.zscore(df[column]))
        df['is_anomaly'] = z_scores > threshold
    elif method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        df['is_anomaly'] = (df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))
    else:
        raise ValueError(f"Unknown anomaly detection method: {method}")
    
    return [CSVRow(row) for _, row in df.iterrows()]