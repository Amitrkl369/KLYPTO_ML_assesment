"""
Feature Engineering Module
Calculate technical indicators, derived features, and transformations
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EMAIndicators:
    """
    Exponential Moving Average indicators
    """
    
    @staticmethod
    def calculate_ema(df: pd.DataFrame, column: str = 'close', period: int = 5) -> pd.Series:
        """
        Calculate Exponential Moving Average
        
        Args:
            df: Input DataFrame
            column: Column to calculate EMA on
            period: EMA period
            
        Returns:
            Series with EMA values
        """
        logger.info(f"Calculating EMA-{period} on {column}")
        return df[column].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def add_ema_indicators(df: pd.DataFrame, fast_period: int = 5, slow_period: int = 15) -> pd.DataFrame:
        """
        Add EMA indicators to DataFrame
        
        Args:
            df: Input DataFrame
            fast_period: Fast EMA period (default: 5)
            slow_period: Slow EMA period (default: 15)
            
        Returns:
            DataFrame with EMA columns added
        """
        df = df.copy()
        df[f'ema_{fast_period}'] = EMAIndicators.calculate_ema(df, 'close', fast_period)
        df[f'ema_{slow_period}'] = EMAIndicators.calculate_ema(df, 'close', slow_period)
        
        # Calculate EMA crossover signals
        df['ema_diff'] = df[f'ema_{fast_period}'] - df[f'ema_{slow_period}']
        df['ema_signal'] = np.where(df['ema_diff'] > 0, 1, -1)
        
        logger.info(f"Added EMA indicators: EMA-{fast_period}, EMA-{slow_period}")
        return df


class DerivedFeatures:
    """
    Calculate derived features from options and futures data
    """
    
    @staticmethod
    def calculate_average_iv(df: pd.DataFrame, call_iv_col: str = 'call_iv', put_iv_col: str = 'put_iv') -> pd.Series:
        """
        Calculate average implied volatility
        
        Args:
            df: Input DataFrame
            call_iv_col: Call IV column name
            put_iv_col: Put IV column name
            
        Returns:
            Series with average IV
        """
        logger.info("Calculating average IV")
        return (df[call_iv_col] + df[put_iv_col]) / 2
    
    @staticmethod
    def calculate_iv_spread(df: pd.DataFrame, call_iv_col: str = 'call_iv', put_iv_col: str = 'put_iv') -> pd.Series:
        """
        Calculate IV spread (call IV - put IV)
        
        Args:
            df: Input DataFrame
            call_iv_col: Call IV column name
            put_iv_col: Put IV column name
            
        Returns:
            Series with IV spread
        """
        logger.info("Calculating IV spread")
        return df[call_iv_col] - df[put_iv_col]
    
    @staticmethod
    def calculate_pcr_oi(df: pd.DataFrame) -> pd.Series:
        """
        Calculate Put-Call Ratio based on Open Interest
        
        Args:
            df: Input DataFrame
            
        Returns:
            Series with PCR (OI-based)
        """
        logger.info("Calculating PCR (OI-based)")
        
        # Sum all put OI and call OI across strikes
        put_oi_cols = [col for col in df.columns if 'put' in col.lower() and 'oi' in col.lower()]
        call_oi_cols = [col for col in df.columns if 'call' in col.lower() and 'oi' in col.lower()]
        
        total_put_oi = df[put_oi_cols].sum(axis=1)
        total_call_oi = df[call_oi_cols].sum(axis=1)
        
        return total_put_oi / (total_call_oi + 1e-10)  # Avoid division by zero
    
    @staticmethod
    def calculate_pcr_volume(df: pd.DataFrame) -> pd.Series:
        """
        Calculate Put-Call Ratio based on Volume
        
        Args:
            df: Input DataFrame
            
        Returns:
            Series with PCR (Volume-based)
        """
        logger.info("Calculating PCR (Volume-based)")
        
        put_vol_cols = [col for col in df.columns if 'put' in col.lower() and 'volume' in col.lower()]
        call_vol_cols = [col for col in df.columns if 'call' in col.lower() and 'volume' in col.lower()]
        
        total_put_volume = df[put_vol_cols].sum(axis=1)
        total_call_volume = df[call_vol_cols].sum(axis=1)
        
        return total_put_volume / (total_call_volume + 1e-10)
    
    @staticmethod
    def calculate_futures_basis(df: pd.DataFrame, futures_col: str = 'close_futures', spot_col: str = 'close_spot') -> pd.Series:
        """
        Calculate futures basis
        
        Args:
            df: Input DataFrame
            futures_col: Futures close price column
            spot_col: Spot close price column
            
        Returns:
            Series with futures basis
        """
        logger.info("Calculating futures basis")
        return (df[futures_col] - df[spot_col]) / df[spot_col]
    
    @staticmethod
    def calculate_returns(df: pd.DataFrame, column: str = 'close', periods: int = 1) -> pd.Series:
        """
        Calculate returns
        
        Args:
            df: Input DataFrame
            column: Column to calculate returns on
            periods: Number of periods for return calculation
            
        Returns:
            Series with returns
        """
        logger.info(f"Calculating returns for {column} with {periods} periods")
        return df[column].pct_change(periods=periods)
    
    @staticmethod
    def calculate_log_returns(df: pd.DataFrame, column: str = 'close') -> pd.Series:
        """
        Calculate log returns
        
        Args:
            df: Input DataFrame
            column: Column to calculate log returns on
            
        Returns:
            Series with log returns
        """
        logger.info(f"Calculating log returns for {column}")
        return np.log(df[column] / df[column].shift(1))
    
    @staticmethod
    def calculate_delta_neutral_ratio(df: pd.DataFrame, call_delta_col: str = 'call_delta', put_delta_col: str = 'put_delta') -> pd.Series:
        """
        Calculate delta neutral ratio
        
        Args:
            df: Input DataFrame
            call_delta_col: Call delta column
            put_delta_col: Put delta column
            
        Returns:
            Series with delta neutral ratio
        """
        logger.info("Calculating delta neutral ratio")
        return np.abs(df[call_delta_col]) / (np.abs(df[put_delta_col]) + 1e-10)
    
    @staticmethod
    def calculate_gamma_exposure(df: pd.DataFrame, spot_col: str = 'close_spot', gamma_col: str = 'gamma', oi_col: str = 'open_interest') -> pd.Series:
        """
        Calculate gamma exposure
        
        Args:
            df: Input DataFrame
            spot_col: Spot price column
            gamma_col: Gamma column
            oi_col: Open interest column
            
        Returns:
            Series with gamma exposure
        """
        logger.info("Calculating gamma exposure")
        return df[spot_col] * df[gamma_col] * df[oi_col]
    
    @staticmethod
    def add_all_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all derived features to DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with all derived features
        """
        logger.info("Adding all derived features")
        
        df = df.copy()
        
        # Average IV
        if 'call_iv' in df.columns and 'put_iv' in df.columns:
            df['avg_iv'] = DerivedFeatures.calculate_average_iv(df)
            df['iv_spread'] = DerivedFeatures.calculate_iv_spread(df)
        
        # PCR
        df['pcr_oi'] = DerivedFeatures.calculate_pcr_oi(df)
        df['pcr_volume'] = DerivedFeatures.calculate_pcr_volume(df)
        
        # Futures basis
        if 'close_futures' in df.columns and 'close_spot' in df.columns:
            df['futures_basis'] = DerivedFeatures.calculate_futures_basis(df)
        
        # Returns
        if 'close_spot' in df.columns:
            df['spot_returns'] = DerivedFeatures.calculate_returns(df, 'close_spot')
            df['spot_log_returns'] = DerivedFeatures.calculate_log_returns(df, 'close_spot')
        
        if 'close_futures' in df.columns:
            df['futures_returns'] = DerivedFeatures.calculate_returns(df, 'close_futures')
        
        # Delta neutral ratio
        if 'call_delta' in df.columns and 'put_delta' in df.columns:
            df['delta_neutral_ratio'] = DerivedFeatures.calculate_delta_neutral_ratio(df)
        
        # Gamma exposure
        if 'close_spot' in df.columns and 'gamma' in df.columns and 'open_interest' in df.columns:
            df['gamma_exposure'] = DerivedFeatures.calculate_gamma_exposure(df)
        
        logger.info("All derived features added successfully")
        return df


class TimeBasedFeatures:
    """
    Time-based feature engineering
    """
    
    @staticmethod
    def add_time_features(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """
        Add time-based features
        
        Args:
            df: Input DataFrame
            timestamp_col: Timestamp column name
            
        Returns:
            DataFrame with time features
        """
        logger.info("Adding time-based features")
        
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        df['hour'] = df[timestamp_col].dt.hour
        df['minute'] = df[timestamp_col].dt.minute
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['day_of_month'] = df[timestamp_col].dt.day
        df['month'] = df[timestamp_col].dt.month
        df['quarter'] = df[timestamp_col].dt.quarter
        
        # Market session features
        df['is_opening_hour'] = (df['hour'] == 9).astype(int)
        df['is_closing_hour'] = (df['hour'] >= 15).astype(int)
        df['is_mid_session'] = ((df['hour'] >= 11) & (df['hour'] <= 14)).astype(int)
        
        logger.info("Time-based features added")
        return df


class LagFeatures:
    """
    Create lag features for time series
    """
    
    @staticmethod
    def add_lag_features(df: pd.DataFrame, columns: list, lags: list = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        Add lag features for specified columns
        
        Args:
            df: Input DataFrame
            columns: List of columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        logger.info(f"Adding lag features for {len(columns)} columns with lags {lags}")
        
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        logger.info("Lag features added")
        return df
    
    @staticmethod
    def add_rolling_features(df: pd.DataFrame, columns: list, windows: list = [5, 10, 20]) -> pd.DataFrame:
        """
        Add rolling statistical features
        
        Args:
            df: Input DataFrame
            columns: List of columns to create rolling features for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        logger.info(f"Adding rolling features for {len(columns)} columns with windows {windows}")
        
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
                    df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window).min()
                    df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window).max()
        
        logger.info("Rolling features added")
        return df


def create_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create complete feature set
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with all features
    """
    logger.info("Creating complete feature set")
    
    # Add EMA indicators
    df = EMAIndicators.add_ema_indicators(df)
    
    # Add derived features
    df = DerivedFeatures.add_all_derived_features(df)
    
    # Add time-based features
    df = TimeBasedFeatures.add_time_features(df)
    
    # Add lag features for key columns
    key_columns = ['close_spot', 'close_futures', 'volume_spot', 'avg_iv', 'pcr_oi']
    df = LagFeatures.add_lag_features(df, key_columns, lags=[1, 2, 3])
    
    logger.info("Complete feature set created")
    return df


def save_features(df: pd.DataFrame, filepath: str):
    """
    Save feature set to CSV
    
    Args:
        df: DataFrame with features
        filepath: Output file path
    """
    df.to_csv(filepath, index=False)
    logger.info(f"Features saved to {filepath}")
