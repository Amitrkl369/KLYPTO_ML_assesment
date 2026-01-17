"""
Trading Strategy Module
Implement EMA crossover strategy with regime filter
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EMAStrategy:
    """
    5/15 EMA Crossover Strategy with Regime Filter
    """
    
    def __init__(self, fast_ema: int = 5, slow_ema: int = 15):
        """
        Initialize EMA strategy
        
        Args:
            fast_ema: Fast EMA period (default: 5)
            slow_ema: Slow EMA period (default: 15)
        """
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        
        logger.info(f"Initialized EMA Strategy: Fast={fast_ema}, Slow={slow_ema}")
    
    def generate_signals(self, df: pd.DataFrame, use_regime_filter: bool = True) -> pd.DataFrame:
        """
        Generate trading signals
        
        Args:
            df: DataFrame with EMA and regime columns
            use_regime_filter: Whether to use regime filter
            
        Returns:
            DataFrame with signal columns
        """
        logger.info("Generating trading signals")
        
        df = df.copy()
        
        # Calculate EMA crossover
        df['ema_cross_up'] = ((df[f'ema_{self.fast_ema}'] > df[f'ema_{self.slow_ema}']) & 
                              (df[f'ema_{self.fast_ema}'].shift(1) <= df[f'ema_{self.slow_ema}'].shift(1)))
        
        df['ema_cross_down'] = ((df[f'ema_{self.fast_ema}'] < df[f'ema_{self.slow_ema}']) & 
                                (df[f'ema_{self.fast_ema}'].shift(1) >= df[f'ema_{self.slow_ema}'].shift(1)))
        
        # Initialize signals
        df['signal'] = 0
        
        if use_regime_filter and 'regime' in df.columns:
            # LONG Entry: EMA cross up AND regime = +1
            df.loc[df['ema_cross_up'] & (df['regime'] == 1), 'signal'] = 1
            
            # SHORT Entry: EMA cross down AND regime = -1
            df.loc[df['ema_cross_down'] & (df['regime'] == -1), 'signal'] = -1
            
            logger.info("Signals generated with regime filter")
        else:
            # Without regime filter
            df.loc[df['ema_cross_up'], 'signal'] = 1
            df.loc[df['ema_cross_down'], 'signal'] = -1
            
            logger.info("Signals generated without regime filter")
        
        return df
    
    def generate_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate positions from signals
        
        Args:
            df: DataFrame with signals
            
        Returns:
            DataFrame with position column
        """
        logger.info("Generating positions from signals")
        
        df = df.copy()
        df['position'] = 0
        
        current_position = 0
        
        for i in range(len(df)):
            signal = df.iloc[i]['signal']
            
            if signal == 1:  # LONG entry
                current_position = 1
            elif signal == -1:  # SHORT entry
                current_position = -1
            elif current_position == 1 and df.iloc[i]['ema_cross_down']:  # LONG exit
                current_position = 0
            elif current_position == -1 and df.iloc[i]['ema_cross_up']:  # SHORT exit
                current_position = 0
            
            df.iloc[i, df.columns.get_loc('position')] = current_position
        
        logger.info(f"Positions generated. Long: {np.sum(df['position']==1)}, "
                   f"Short: {np.sum(df['position']==-1)}, Flat: {np.sum(df['position']==0)}")
        
        return df


class TradeAnalyzer:
    """
    Analyze individual trades
    """
    
    @staticmethod
    def extract_trades(df: pd.DataFrame, price_col: str = 'close_spot') -> pd.DataFrame:
        """
        Extract individual trades from position data
        
        Args:
            df: DataFrame with positions
            price_col: Price column to use
            
        Returns:
            DataFrame with trade details
        """
        logger.info("Extracting individual trades")
        
        trades = []
        in_trade = False
        trade_entry = None
        
        for i in range(len(df)):
            position = df.iloc[i]['position']
            prev_position = df.iloc[i-1]['position'] if i > 0 else 0
            
            # Trade entry
            if position != 0 and prev_position == 0:
                in_trade = True
                trade_entry = {
                    'entry_idx': i,
                    'entry_time': df.iloc[i]['timestamp'] if 'timestamp' in df.columns else i,
                    'entry_price': df.iloc[i][price_col],
                    'direction': 'LONG' if position == 1 else 'SHORT',
                    'regime': df.iloc[i]['regime'] if 'regime' in df.columns else None
                }
            
            # Trade exit
            elif position == 0 and prev_position != 0 and in_trade:
                trade_entry['exit_idx'] = i
                trade_entry['exit_time'] = df.iloc[i]['timestamp'] if 'timestamp' in df.columns else i
                trade_entry['exit_price'] = df.iloc[i][price_col]
                
                # Calculate PnL
                if trade_entry['direction'] == 'LONG':
                    trade_entry['pnl'] = trade_entry['exit_price'] - trade_entry['entry_price']
                else:  # SHORT
                    trade_entry['pnl'] = trade_entry['entry_price'] - trade_entry['exit_price']
                
                trade_entry['pnl_pct'] = (trade_entry['pnl'] / trade_entry['entry_price']) * 100
                trade_entry['duration'] = trade_entry['exit_idx'] - trade_entry['entry_idx']
                
                trades.append(trade_entry)
                in_trade = False
                trade_entry = None
        
        trades_df = pd.DataFrame(trades)
        
        if len(trades_df) > 0:
            trades_df['is_profitable'] = trades_df['pnl'] > 0
            logger.info(f"Extracted {len(trades_df)} trades")
        else:
            logger.warning("No trades extracted")
        
        return trades_df
    
    @staticmethod
    def calculate_trade_statistics(trades_df: pd.DataFrame) -> dict:
        """
        Calculate trade statistics
        
        Args:
            trades_df: DataFrame with trade details
            
        Returns:
            Dictionary with trade statistics
        """
        if len(trades_df) == 0:
            return {}
        
        logger.info("Calculating trade statistics")
        
        profitable_trades = trades_df[trades_df['is_profitable']]
        losing_trades = trades_df[~trades_df['is_profitable']]
        
        stats = {
            'total_trades': len(trades_df),
            'winning_trades': len(profitable_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(profitable_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
            'avg_win': profitable_trades['pnl'].mean() if len(profitable_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'largest_win': profitable_trades['pnl'].max() if len(profitable_trades) > 0 else 0,
            'largest_loss': losing_trades['pnl'].min() if len(losing_trades) > 0 else 0,
            'avg_duration': trades_df['duration'].mean(),
            'total_pnl': trades_df['pnl'].sum(),
            'avg_pnl': trades_df['pnl'].mean(),
        }
        
        # Profit factor
        gross_profit = profitable_trades['pnl'].sum() if len(profitable_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        stats['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Regime breakdown
        if 'regime' in trades_df.columns:
            stats['trades_by_regime'] = trades_df['regime'].value_counts().to_dict()
        
        logger.info(f"Trade statistics calculated. Win rate: {stats['win_rate']:.2f}%")
        
        return stats


class StrategyOptimizer:
    """
    Optimize strategy parameters
    """
    
    @staticmethod
    def grid_search(df: pd.DataFrame, fast_ema_range: List[int], slow_ema_range: List[int],
                   use_regime_filter: bool = True) -> pd.DataFrame:
        """
        Grid search for optimal EMA parameters
        
        Args:
            df: Input DataFrame
            fast_ema_range: Range of fast EMA periods to test
            slow_ema_range: Range of slow EMA periods to test
            use_regime_filter: Whether to use regime filter
            
        Returns:
            DataFrame with optimization results
        """
        from .backtest import Backtester
        
        logger.info(f"Starting grid search: Fast EMA {fast_ema_range}, Slow EMA {slow_ema_range}")
        
        results = []
        
        for fast in fast_ema_range:
            for slow in slow_ema_range:
                if fast >= slow:
                    continue
                
                # Calculate EMAs
                df_test = df.copy()
                df_test[f'ema_{fast}'] = df_test['close_spot'].ewm(span=fast, adjust=False).mean()
                df_test[f'ema_{slow}'] = df_test['close_spot'].ewm(span=slow, adjust=False).mean()
                
                # Generate signals
                strategy = EMAStrategy(fast_ema=fast, slow_ema=slow)
                df_test = strategy.generate_signals(df_test, use_regime_filter)
                df_test = strategy.generate_positions(df_test)
                
                # Backtest
                backtester = Backtester()
                metrics = backtester.calculate_metrics(df_test)
                
                results.append({
                    'fast_ema': fast,
                    'slow_ema': slow,
                    'total_return': metrics.get('total_return', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'total_trades': metrics.get('total_trades', 0)
                })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)
        
        logger.info(f"Grid search completed. Best Sharpe: {results_df.iloc[0]['sharpe_ratio']:.2f}")
        
        return results_df


def apply_ml_filter(df: pd.DataFrame, ml_predictions: np.ndarray, 
                   confidence_threshold: float = 0.5) -> pd.DataFrame:
    """
    Apply ML model filter to trading signals
    
    Args:
        df: DataFrame with signals
        ml_predictions: ML model predictions (probability of profitable trade)
        confidence_threshold: Minimum confidence to take trade
        
    Returns:
        DataFrame with filtered signals
    """
    logger.info(f"Applying ML filter with confidence threshold: {confidence_threshold}")
    
    df = df.copy()
    
    # Only take trades where ML predicts profitable with high confidence
    df['ml_prediction'] = ml_predictions
    df['signal_filtered'] = df['signal'] * (ml_predictions > confidence_threshold).astype(int)
    
    # Regenerate positions with filtered signals
    df_filtered = df.copy()
    df_filtered['signal'] = df_filtered['signal_filtered']
    
    strategy = EMAStrategy()
    df_filtered = strategy.generate_positions(df_filtered)
    
    logger.info(f"ML filter applied. Original signals: {np.sum(df['signal']!=0)}, "
               f"Filtered signals: {np.sum(df_filtered['signal']!=0)}")
    
    return df_filtered


def save_strategy_signals(df: pd.DataFrame, filepath: str):
    """
    Save strategy signals to CSV
    
    Args:
        df: DataFrame with signals
        filepath: Output file path
    """
    signal_cols = ['timestamp', 'close_spot', f'ema_5', f'ema_15', 
                  'regime', 'signal', 'position']
    
    # Filter to columns that exist
    cols_to_save = [col for col in signal_cols if col in df.columns]
    
    df[cols_to_save].to_csv(filepath, index=False)
    logger.info(f"Strategy signals saved to {filepath}")
