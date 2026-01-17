"""
Backtesting Module
Calculate performance metrics and backtest trading strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    """
    Backtest trading strategies and calculate performance metrics
    """
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.0003):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting capital
            commission: Commission per trade (as fraction, e.g., 0.0003 = 0.03%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        
        logger.info(f"Initialized Backtester with capital={initial_capital}, commission={commission}")
    
    def calculate_returns(self, df: pd.DataFrame, price_col: str = 'close_spot') -> pd.DataFrame:
        """
        Calculate strategy returns
        
        Args:
            df: DataFrame with positions
            price_col: Price column to use
            
        Returns:
            DataFrame with returns columns
        """
        logger.info("Calculating strategy returns")
        
        df = df.copy()
        
        # Calculate market returns
        df['market_returns'] = df[price_col].pct_change()
        
        # Calculate strategy returns
        df['strategy_returns'] = df['position'].shift(1) * df['market_returns']
        
        # Apply commission on position changes
        df['position_change'] = df['position'].diff().abs()
        df['commission_cost'] = df['position_change'] * self.commission
        df['strategy_returns'] = df['strategy_returns'] - df['commission_cost']
        
        # Calculate cumulative returns
        df['cumulative_market_returns'] = (1 + df['market_returns']).cumprod()
        df['cumulative_strategy_returns'] = (1 + df['strategy_returns']).cumprod()
        
        # Calculate equity curve
        df['equity'] = self.initial_capital * df['cumulative_strategy_returns']
        
        logger.info("Returns calculated")
        return df
    
    def calculate_metrics(self, df: pd.DataFrame, risk_free_rate: float = 0.065) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Args:
            df: DataFrame with returns
            risk_free_rate: Annual risk-free rate (default: 6.5%)
            
        Returns:
            Dictionary with performance metrics
        """
        logger.info("Calculating performance metrics")
        
        # Calculate returns if not already present
        if 'strategy_returns' not in df.columns:
            df = self.calculate_returns(df)
        
        metrics = {}
        
        # Total return
        total_return = (df['cumulative_strategy_returns'].iloc[-1] - 1) * 100
        metrics['total_return'] = total_return
        
        # Annualized return (assuming 252 trading days per year)
        n_periods = len(df)
        years = n_periods / (252 * 75)  # 75 five-minute periods per day
        annualized_return = ((df['cumulative_strategy_returns'].iloc[-1]) ** (1/years) - 1) * 100 if years > 0 else 0
        metrics['annualized_return'] = annualized_return
        
        # Volatility (annualized)
        daily_returns = df['strategy_returns'].resample('D').sum() if 'timestamp' in df.columns else df['strategy_returns']
        volatility = daily_returns.std() * np.sqrt(252) * 100
        metrics['volatility'] = volatility
        
        # Sharpe Ratio
        excess_returns = daily_returns.mean() * 252 - risk_free_rate
        sharpe_ratio = excess_returns / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
        metrics['sharpe_ratio'] = sharpe_ratio
        
        # Sortino Ratio
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = excess_returns / downside_std if downside_std > 0 else 0
        metrics['sortino_ratio'] = sortino_ratio
        
        # Maximum Drawdown
        cumulative = df['cumulative_strategy_returns']
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        metrics['max_drawdown'] = max_drawdown
        
        # Calmar Ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        metrics['calmar_ratio'] = calmar_ratio
        
        # Trade statistics
        from .strategy import TradeAnalyzer
        trades_df = TradeAnalyzer.extract_trades(df)
        
        if len(trades_df) > 0:
            trade_stats = TradeAnalyzer.calculate_trade_statistics(trades_df)
            metrics.update(trade_stats)
        else:
            metrics['total_trades'] = 0
            metrics['win_rate'] = 0
            metrics['profit_factor'] = 0
        
        logger.info(f"Metrics calculated. Sharpe: {sharpe_ratio:.2f}, Max DD: {max_drawdown:.2f}%")
        
        return metrics
    
    def split_data(self, df: pd.DataFrame, train_size: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets
        
        Args:
            df: Input DataFrame
            train_size: Fraction of data for training
            
        Returns:
            Tuple of (train_df, test_df)
        """
        split_idx = int(len(df) * train_size)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        logger.info(f"Data split: Train={len(train_df)}, Test={len(test_df)}")
        
        return train_df, test_df
    
    def backtest(self, df: pd.DataFrame, strategy_name: str = "Strategy") -> Dict:
        """
        Run complete backtest
        
        Args:
            df: DataFrame with positions
            strategy_name: Name of strategy
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running backtest for {strategy_name}")
        
        # Calculate returns
        df = self.calculate_returns(df)
        
        # Calculate metrics
        metrics = self.calculate_metrics(df)
        
        results = {
            'strategy_name': strategy_name,
            'metrics': metrics,
            'equity_curve': df[['timestamp', 'equity']].copy() if 'timestamp' in df.columns else None,
            'trades': None
        }
        
        # Extract trades
        from .strategy import TradeAnalyzer
        trades_df = TradeAnalyzer.extract_trades(df)
        if len(trades_df) > 0:
            results['trades'] = trades_df
        
        logger.info(f"Backtest completed for {strategy_name}")
        
        return results
    
    def compare_strategies(self, results_list: list) -> pd.DataFrame:
        """
        Compare multiple strategy results
        
        Args:
            results_list: List of backtest results dictionaries
            
        Returns:
            DataFrame with comparison
        """
        logger.info(f"Comparing {len(results_list)} strategies")
        
        comparison_data = []
        
        for result in results_list:
            metrics = result['metrics']
            comparison_data.append({
                'Strategy': result['strategy_name'],
                'Total Return (%)': metrics.get('total_return', 0),
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                'Sortino Ratio': metrics.get('sortino_ratio', 0),
                'Calmar Ratio': metrics.get('calmar_ratio', 0),
                'Max Drawdown (%)': metrics.get('max_drawdown', 0),
                'Win Rate (%)': metrics.get('win_rate', 0),
                'Profit Factor': metrics.get('profit_factor', 0),
                'Total Trades': metrics.get('total_trades', 0),
                'Avg Duration': metrics.get('avg_duration', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        logger.info("Strategy comparison completed")
        
        return comparison_df


class BacktestVisualizer:
    """
    Visualization tools for backtest results
    """
    
    @staticmethod
    def plot_equity_curve(df: pd.DataFrame, save_path: str = None):
        """
        Plot equity curve
        
        Args:
            df: DataFrame with equity column
            save_path: Path to save plot
        """
        logger.info("Plotting equity curve")
        
        plt.figure(figsize=(15, 7))
        
        if 'timestamp' in df.columns:
            plt.plot(df['timestamp'], df['equity'], label='Strategy Equity', linewidth=2)
            plt.xlabel('Timestamp')
        else:
            plt.plot(df['equity'], label='Strategy Equity', linewidth=2)
            plt.xlabel('Period')
        
        plt.ylabel('Equity ($)')
        plt.title('Strategy Equity Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_drawdown(df: pd.DataFrame, save_path: str = None):
        """
        Plot drawdown chart
        
        Args:
            df: DataFrame with cumulative returns
            save_path: Path to save plot
        """
        logger.info("Plotting drawdown")
        
        cumulative = df['cumulative_strategy_returns']
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max * 100
        
        plt.figure(figsize=(15, 5))
        
        if 'timestamp' in df.columns:
            plt.fill_between(df['timestamp'], drawdown, 0, alpha=0.3, color='red')
            plt.plot(df['timestamp'], drawdown, color='red', linewidth=1)
            plt.xlabel('Timestamp')
        else:
            plt.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
            plt.plot(drawdown, color='red', linewidth=1)
            plt.xlabel('Period')
        
        plt.ylabel('Drawdown (%)')
        plt.title('Strategy Drawdown')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_returns_distribution(df: pd.DataFrame, save_path: str = None):
        """
        Plot returns distribution
        
        Args:
            df: DataFrame with strategy returns
            save_path: Path to save plot
        """
        logger.info("Plotting returns distribution")
        
        plt.figure(figsize=(12, 5))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(df['strategy_returns'].dropna() * 100, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Returns (%)')
        plt.ylabel('Frequency')
        plt.title('Returns Distribution')
        plt.grid(True, alpha=0.3)
        
        # Q-Q plot
        plt.subplot(1, 2, 2)
        from scipy import stats
        stats.probplot(df['strategy_returns'].dropna(), dist="norm", plot=plt)
        plt.title('Q-Q Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_trade_analysis(trades_df: pd.DataFrame, save_path: str = None):
        """
        Plot trade analysis
        
        Args:
            trades_df: DataFrame with trade details
            save_path: Path to save plot
        """
        logger.info("Plotting trade analysis")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # PnL distribution
        axes[0, 0].hist(trades_df['pnl'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('PnL')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Trade PnL Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Duration distribution
        axes[0, 1].hist(trades_df['duration'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Duration (periods)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Trade Duration Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # PnL vs Duration scatter
        colors = ['green' if p > 0 else 'red' for p in trades_df['pnl']]
        axes[1, 0].scatter(trades_df['duration'], trades_df['pnl'], c=colors, alpha=0.6)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[1, 0].set_xlabel('Duration (periods)')
        axes[1, 0].set_ylabel('PnL')
        axes[1, 0].set_title('PnL vs Duration')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative PnL
        axes[1, 1].plot(range(len(trades_df)), trades_df['pnl'].cumsum(), linewidth=2)
        axes[1, 1].set_xlabel('Trade Number')
        axes[1, 1].set_ylabel('Cumulative PnL')
        axes[1, 1].set_title('Cumulative PnL Over Trades')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_strategy_comparison(comparison_df: pd.DataFrame, save_path: str = None):
        """
        Plot strategy comparison
        
        Args:
            comparison_df: DataFrame with strategy comparison
            save_path: Path to save plot
        """
        logger.info("Plotting strategy comparison")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        metrics = ['Total Return (%)', 'Sharpe Ratio', 'Sortino Ratio', 
                  'Max Drawdown (%)', 'Win Rate (%)', 'Profit Factor']
        
        for idx, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                ax = axes[idx]
                comparison_df.plot(x='Strategy', y=metric, kind='bar', ax=ax, legend=False)
                ax.set_title(metric)
                ax.set_xlabel('')
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.close()


def generate_backtest_report(results: Dict, output_path: str):
    """
    Generate comprehensive backtest report
    
    Args:
        results: Backtest results dictionary
        output_path: Path to save report
    """
    logger.info("Generating backtest report")
    
    report = []
    report.append("=" * 80)
    report.append(f"BACKTEST REPORT: {results['strategy_name']}")
    report.append("=" * 80)
    report.append("")
    
    metrics = results['metrics']
    
    report.append("PERFORMANCE METRICS")
    report.append("-" * 80)
    report.append(f"Total Return: {metrics.get('total_return', 0):.2f}%")
    report.append(f"Annualized Return: {metrics.get('annualized_return', 0):.2f}%")
    report.append(f"Volatility: {metrics.get('volatility', 0):.2f}%")
    report.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    report.append(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
    report.append(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
    report.append(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
    report.append("")
    
    report.append("TRADE STATISTICS")
    report.append("-" * 80)
    report.append(f"Total Trades: {metrics.get('total_trades', 0)}")
    report.append(f"Winning Trades: {metrics.get('winning_trades', 0)}")
    report.append(f"Losing Trades: {metrics.get('losing_trades', 0)}")
    report.append(f"Win Rate: {metrics.get('win_rate', 0):.2f}%")
    report.append(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    report.append(f"Average Win: {metrics.get('avg_win', 0):.2f}")
    report.append(f"Average Loss: {metrics.get('avg_loss', 0):.2f}")
    report.append(f"Largest Win: {metrics.get('largest_win', 0):.2f}")
    report.append(f"Largest Loss: {metrics.get('largest_loss', 0):.2f}")
    report.append(f"Average Duration: {metrics.get('avg_duration', 0):.2f} periods")
    report.append("")
    
    if 'trades_by_regime' in metrics:
        report.append("TRADES BY REGIME")
        report.append("-" * 80)
        for regime, count in metrics['trades_by_regime'].items():
            regime_name = {-1: 'Downtrend', 0: 'Sideways', 1: 'Uptrend'}.get(regime, regime)
            report.append(f"{regime_name}: {count} trades")
        report.append("")
    
    report.append("=" * 80)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    logger.info(f"Backtest report saved to {output_path}")


def save_backtest_results(results: Dict, base_path: str):
    """
    Save backtest results to files
    
    Args:
        results: Backtest results dictionary
        base_path: Base path for saving files
    """
    import os
    
    # Save equity curve
    if results['equity_curve'] is not None:
        results['equity_curve'].to_csv(f"{base_path}_equity_curve.csv", index=False)
    
    # Save trades
    if results['trades'] is not None:
        results['trades'].to_csv(f"{base_path}_trades.csv", index=False)
    
    # Save metrics
    metrics_df = pd.DataFrame([results['metrics']])
    metrics_df.to_csv(f"{base_path}_metrics.csv", index=False)
    
    logger.info(f"Backtest results saved to {base_path}_*")
