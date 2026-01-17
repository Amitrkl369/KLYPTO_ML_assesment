"""
Data Utilities Module
Handles data fetching, cleaning, merging, and preprocessing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Fetch data from various sources (Zerodha, ICICI Breeze, NSE, etc.)
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize DataFetcher with API credentials
        
        Args:
            api_key: API key for data source
            api_secret: API secret for data source
        """
        self.api_key = api_key
        self.api_secret = api_secret
        
    def fetch_nifty_spot(self, start_date: str, end_date: str, interval: str = '5minute') -> pd.DataFrame:
        """
        Fetch NIFTY 50 Spot OHLCV data
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Time interval (default: 5minute)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        logger.info(f"Fetching NIFTY Spot data from {start_date} to {end_date}")
        # Implementation would use actual API calls
        # Placeholder for demonstration
        return pd.DataFrame()
    
    def fetch_nifty_futures(self, start_date: str, end_date: str, interval: str = '5minute') -> pd.DataFrame:
        """
        Fetch NIFTY Futures data with contract rollover handling
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Time interval (default: 5minute)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, open_interest
        """
        logger.info(f"Fetching NIFTY Futures data from {start_date} to {end_date}")
        return pd.DataFrame()
    
    def fetch_nifty_options(self, start_date: str, end_date: str, interval: str = '5minute') -> pd.DataFrame:
        """
        Fetch NIFTY Options Chain data (ATM ± 2 strikes)
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Time interval (default: 5minute)
            
        Returns:
            DataFrame with option chain data
        """
        logger.info(f"Fetching NIFTY Options data from {start_date} to {end_date}")
        return pd.DataFrame()
    
    def calculate_atm_strike(self, spot_price: float, strike_gap: int = 50) -> int:
        """
        Calculate ATM strike dynamically
        
        Args:
            spot_price: Current spot price
            strike_gap: Gap between strikes (default: 50)
            
        Returns:
            ATM strike price
        """
        return round(spot_price / strike_gap) * strike_gap


class DataCleaner:
    """
    Clean and preprocess market data
    """
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            method: Method to handle missing values ('ffill', 'bfill', 'interpolate', 'drop')
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Handling missing values using method: {method}")
        
        if method == 'ffill':
            df = df.fillna(method='ffill')
        elif method == 'bfill':
            df = df.fillna(method='bfill')
        elif method == 'interpolate':
            df = df.interpolate(method='linear')
        elif method == 'drop':
            df = df.dropna()
            
        return df
    
    @staticmethod
    def remove_outliers(df: pd.DataFrame, columns: List[str], method: str = 'zscore', threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers from specified columns
        
        Args:
            df: Input DataFrame
            columns: List of columns to check for outliers
            method: Method for outlier detection ('zscore', 'iqr')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        logger.info(f"Removing outliers using {method} method with threshold {threshold}")
        
        df_clean = df.copy()
        
        for col in columns:
            if method == 'zscore':
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                df_clean = df_clean[z_scores < threshold]
            elif method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        logger.info(f"Removed {len(df) - len(df_clean)} outlier rows")
        return df_clean
    
    @staticmethod
    def align_timestamps(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """
        Align timestamps across multiple DataFrames
        
        Args:
            dfs: List of DataFrames to align
            
        Returns:
            List of aligned DataFrames
        """
        logger.info("Aligning timestamps across datasets")
        
        # Find common timestamps
        common_timestamps = set(dfs[0]['timestamp'])
        for df in dfs[1:]:
            common_timestamps &= set(df['timestamp'])
        
        # Filter to common timestamps
        aligned_dfs = []
        for df in dfs:
            aligned_df = df[df['timestamp'].isin(common_timestamps)].copy()
            aligned_dfs.append(aligned_df)
        
        logger.info(f"Aligned to {len(common_timestamps)} common timestamps")
        return aligned_dfs
    
    @staticmethod
    def handle_futures_rollover(df: pd.DataFrame, rollover_days: int = 5) -> pd.DataFrame:
        """
        Handle futures contract rollover
        
        Args:
            df: Futures DataFrame
            rollover_days: Days before expiry to roll over
            
        Returns:
            DataFrame with adjusted prices for rollover
        """
        logger.info("Handling futures contract rollover")
        # Implementation for contract rollover logic
        return df


class DataMerger:
    """
    Merge spot, futures, and options data
    """
    
    @staticmethod
    def merge_datasets(spot_df: pd.DataFrame, futures_df: pd.DataFrame, options_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge spot, futures, and options data on timestamp
        
        Args:
            spot_df: Spot data DataFrame
            futures_df: Futures data DataFrame
            options_df: Options data DataFrame
            
        Returns:
            Merged DataFrame
        """
        logger.info("Merging spot, futures, and options data")
        
        # Merge spot and futures
        merged = pd.merge(spot_df, futures_df, on='timestamp', suffixes=('_spot', '_futures'))
        
        # Merge with options
        merged = pd.merge(merged, options_df, on='timestamp', how='inner')
        
        logger.info(f"Merged dataset has {len(merged)} rows")
        return merged
    
    @staticmethod
    def save_cleaned_data(df: pd.DataFrame, filepath: str):
        """
        Save cleaned data to CSV
        
        Args:
            df: DataFrame to save
            filepath: Output file path
        """
        df.to_csv(filepath, index=False)
        logger.info(f"Saved data to {filepath}")
    
    @staticmethod
    def generate_cleaning_report(original_df: pd.DataFrame, cleaned_df: pd.DataFrame, filepath: str):
        """
        Generate data cleaning report
        
        Args:
            original_df: Original DataFrame
            cleaned_df: Cleaned DataFrame
            filepath: Output file path for report
        """
        report = []
        report.append("=" * 80)
        report.append("DATA CLEANING REPORT")
        report.append("=" * 80)
        report.append(f"\nOriginal dataset rows: {len(original_df)}")
        report.append(f"Cleaned dataset rows: {len(cleaned_df)}")
        report.append(f"Rows removed: {len(original_df) - len(cleaned_df)}")
        report.append(f"Removal percentage: {((len(original_df) - len(cleaned_df)) / len(original_df) * 100):.2f}%")
        
        report.append("\n" + "=" * 80)
        report.append("MISSING VALUES SUMMARY")
        report.append("=" * 80)
        report.append("\nOriginal dataset:")
        report.append(str(original_df.isnull().sum()))
        report.append("\nCleaned dataset:")
        report.append(str(cleaned_df.isnull().sum()))
        
        report.append("\n" + "=" * 80)
        report.append("STATISTICAL SUMMARY")
        report.append("=" * 80)
        report.append("\nCleaned dataset statistics:")
        report.append(str(cleaned_df.describe()))
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Generated cleaning report: {filepath}")


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV file
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with loaded data
    """
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    
    # Convert timestamp to datetime if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate data quality
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if data is valid, False otherwise
    """
    # Check for required columns
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Check for negative prices
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        if (df[col] < 0).any():
            logger.error(f"Negative values found in {col}")
            return False
    
    # Check OHLC consistency
    if ((df['high'] < df['low']) | 
        (df['high'] < df['open']) | 
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])).any():
        logger.error("OHLC inconsistency detected")
        return False
    
    logger.info("Data validation passed")
    return True
