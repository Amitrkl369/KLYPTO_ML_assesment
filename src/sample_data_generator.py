"""
Generate Sample/Mock Market Data for Testing
Creates realistic NIFTY 50 OHLCV data without needing API keys
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_nifty_spot_data(start_date, end_date, interval_minutes=5):
    """
    Generate realistic NIFTY 50 spot data
    
    Args:
        start_date: Start date (datetime)
        end_date: End date (datetime)
        interval_minutes: Interval in minutes (default: 5)
    
    Returns:
        DataFrame with OHLCV data
    """
    # Generate timestamps
    timestamps = []
    current = start_date
    while current <= end_date:
        # Only trading hours (9:15 AM to 3:30 PM IST)
        if current.weekday() < 5 and 9 <= current.hour <= 15:
            timestamps.append(current)
        current += timedelta(minutes=interval_minutes)
    
    # Starting price
    base_price = 18000
    prices = []
    
    # Generate realistic price movements using geometric Brownian motion
    np.random.seed(42)
    for i, ts in enumerate(timestamps):
        if i == 0:
            close = base_price
        else:
            # Random walk with drift
            daily_return = np.random.normal(0.0005, 0.01)
            close = prices[-1]['close'] * (1 + daily_return)
        
        # Generate OHLC
        open_price = close * np.random.uniform(0.9995, 1.0005)
        high = max(open_price, close) * np.random.uniform(1, 1.002)
        low = min(open_price, close) * np.random.uniform(0.998, 1)
        volume = np.random.randint(100000, 5000000)
        
        prices.append({
            'timestamp': ts,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })
    
    df = pd.DataFrame(prices)
    return df

def generate_nifty_futures_data(spot_df):
    """
    Generate NIFTY Futures data (based on spot with futures premium)
    
    Args:
        spot_df: NIFTY spot data DataFrame
    
    Returns:
        DataFrame with futures OHLCV + Open Interest
    """
    futures_df = spot_df.copy()
    
    # Add futures premium (typically 0.2-0.5% above spot)
    premium = np.random.uniform(0.002, 0.005)
    futures_df['open'] = (futures_df['open'] * (1 + premium)).round(2)
    futures_df['high'] = (futures_df['high'] * (1 + premium)).round(2)
    futures_df['low'] = (futures_df['low'] * (1 + premium)).round(2)
    futures_df['close'] = (futures_df['close'] * (1 + premium)).round(2)
    
    # Add open interest (grows over time)
    futures_df['open_interest'] = np.linspace(100000, 500000, len(futures_df)).astype(int)
    
    return futures_df

def generate_nifty_options_data(spot_df):
    """
    Generate NIFTY Options Chain data (ATM ± 2 strikes)
    
    Args:
        spot_df: NIFTY spot data DataFrame
    
    Returns:
        DataFrame with options data
    """
    options_data = []
    
    for _, row in spot_df.iterrows():
        spot = row['close']
        strike_gap = 50
        atm_strike = round(spot / strike_gap) * strike_gap
        
        # Create ATM ± 2 strikes
        for strike_offset in [-100, -50, 0, 50, 100]:
            strike = atm_strike + strike_offset
            
            # Call option
            call_iv = np.random.uniform(0.15, 0.35)
            call_premium = max(spot - strike, 0) * 0.05 + np.random.uniform(5, 50)
            
            # Put option
            put_iv = np.random.uniform(0.15, 0.35)
            put_premium = max(strike - spot, 0) * 0.05 + np.random.uniform(5, 50)
            
            options_data.append({
                'timestamp': row['timestamp'],
                'strike': strike,
                'call_premium': round(call_premium, 2),
                'call_iv': round(call_iv, 4),
                'call_volume': np.random.randint(10000, 500000),
                'put_premium': round(put_premium, 2),
                'put_iv': round(put_iv, 4),
                'put_volume': np.random.randint(10000, 500000),
                'spot': spot
            })
    
    return pd.DataFrame(options_data)

def save_sample_data(output_dir='../data'):
    """
    Generate and save all sample data
    
    Args:
        output_dir: Directory to save CSV files
    """
    print("Generating sample market data...")
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Generate data
    spot_df = generate_nifty_spot_data(start_date, end_date)
    futures_df = generate_nifty_futures_data(spot_df)
    options_df = generate_nifty_options_data(spot_df)
    
    # Create directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    spot_df.to_csv(f'{output_dir}/nifty_spot.csv', index=False)
    futures_df.to_csv(f'{output_dir}/nifty_futures.csv', index=False)
    options_df.to_csv(f'{output_dir}/nifty_options.csv', index=False)
    
    print(f"✓ Spot data: {len(spot_df)} rows")
    print(f"✓ Futures data: {len(futures_df)} rows")
    print(f"✓ Options data: {len(options_df)} rows")
    print(f"✓ Data saved to {output_dir}/")
    
    return spot_df, futures_df, options_df

if __name__ == '__main__':
    save_sample_data()
