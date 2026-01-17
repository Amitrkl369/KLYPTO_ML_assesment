"""
Regime Detection Module
Implement Hidden Markov Model for market regime classification
"""

import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Hidden Markov Model for market regime detection
    """
    
    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        """
        Initialize regime detector
        
        Args:
            n_regimes: Number of market regimes (default: 3)
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        logger.info(f"Initialized RegimeDetector with {n_regimes} regimes")
    
    def prepare_features(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """
        Prepare and normalize features for HMM
        
        Args:
            df: Input DataFrame
            feature_cols: List of feature column names
            
        Returns:
            Normalized feature array
        """
        from sklearn.preprocessing import StandardScaler
        
        logger.info(f"Preparing {len(feature_cols)} features for HMM")
        
        # Extract features
        X = df[feature_cols].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize features
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        self.feature_names = feature_cols
        logger.info("Features prepared and normalized")
        
        return X_scaled
    
    def fit(self, df: pd.DataFrame, feature_cols: List[str], n_iter: int = 100):
        """
        Fit HMM to training data
        
        Args:
            df: Training DataFrame
            feature_cols: List of feature columns to use
            n_iter: Number of EM iterations
        """
        logger.info("Fitting HMM to training data")
        
        # Prepare features
        X = self.prepare_features(df, feature_cols)
        
        # Initialize and fit Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type='full',
            n_iter=n_iter,
            random_state=self.random_state,
            verbose=False
        )
        
        self.model.fit(X)
        
        logger.info(f"HMM fitted successfully. Log-likelihood: {self.model.score(X):.2f}")
    
    def predict(self, df: pd.DataFrame, feature_cols: List[str] = None) -> np.ndarray:
        """
        Predict regimes for new data
        
        Args:
            df: Input DataFrame
            feature_cols: List of feature columns (if None, uses training features)
            
        Returns:
            Array of predicted regimes
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if feature_cols is None:
            feature_cols = self.feature_names
        
        logger.info("Predicting regimes")
        
        # Prepare features
        X = self.prepare_features(df, feature_cols)
        
        # Predict regimes
        regimes = self.model.predict(X)
        
        return regimes
    
    def map_regimes_to_labels(self, df: pd.DataFrame, regimes: np.ndarray, returns_col: str = 'spot_returns') -> np.ndarray:
        """
        Map numeric regimes to semantic labels (+1: Uptrend, -1: Downtrend, 0: Sideways)
        
        Args:
            df: DataFrame with returns
            regimes: Array of numeric regimes (0, 1, 2)
            returns_col: Column name for returns
            
        Returns:
            Array of labeled regimes (-1, 0, +1)
        """
        logger.info("Mapping regimes to semantic labels")
        
        # Calculate average return for each regime
        regime_returns = {}
        for regime in range(self.n_regimes):
            mask = regimes == regime
            avg_return = df.loc[mask, returns_col].mean()
            regime_returns[regime] = avg_return
        
        # Sort regimes by average return
        sorted_regimes = sorted(regime_returns.items(), key=lambda x: x[1])
        
        # Map: lowest return = -1 (downtrend), middle = 0 (sideways), highest = +1 (uptrend)
        regime_mapping = {
            sorted_regimes[0][0]: -1,  # Downtrend
            sorted_regimes[1][0]: 0,   # Sideways
            sorted_regimes[2][0]: 1    # Uptrend
        }
        
        # Apply mapping
        labeled_regimes = np.array([regime_mapping[r] for r in regimes])
        
        logger.info(f"Regime mapping: {regime_mapping}")
        logger.info(f"Regime distribution: Downtrend={np.sum(labeled_regimes==-1)}, "
                   f"Sideways={np.sum(labeled_regimes==0)}, Uptrend={np.sum(labeled_regimes==1)}")
        
        return labeled_regimes
    
    def get_transition_matrix(self) -> np.ndarray:
        """
        Get regime transition probability matrix
        
        Returns:
            Transition matrix
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.transmat_
    
    def get_regime_statistics(self, df: pd.DataFrame, regimes: np.ndarray, feature_cols: List[str]) -> pd.DataFrame:
        """
        Calculate statistics for each regime
        
        Args:
            df: Input DataFrame
            regimes: Array of regime labels
            feature_cols: Features to analyze
            
        Returns:
            DataFrame with regime statistics
        """
        logger.info("Calculating regime statistics")
        
        stats_list = []
        
        for regime in sorted(np.unique(regimes)):
            regime_data = df[regimes == regime]
            
            stats = {
                'regime': regime,
                'count': len(regime_data),
                'percentage': len(regime_data) / len(df) * 100
            }
            
            # Calculate mean and std for each feature
            for col in feature_cols:
                if col in df.columns:
                    stats[f'{col}_mean'] = regime_data[col].mean()
                    stats[f'{col}_std'] = regime_data[col].std()
            
            stats_list.append(stats)
        
        stats_df = pd.DataFrame(stats_list)
        logger.info("Regime statistics calculated")
        
        return stats_df
    
    def calculate_regime_durations(self, regimes: np.ndarray) -> dict:
        """
        Calculate duration statistics for each regime
        
        Args:
            regimes: Array of regime labels
            
        Returns:
            Dictionary with duration statistics
        """
        logger.info("Calculating regime durations")
        
        durations = {regime: [] for regime in np.unique(regimes)}
        
        current_regime = regimes[0]
        duration = 1
        
        for i in range(1, len(regimes)):
            if regimes[i] == current_regime:
                duration += 1
            else:
                durations[current_regime].append(duration)
                current_regime = regimes[i]
                duration = 1
        
        # Add last duration
        durations[current_regime].append(duration)
        
        # Calculate statistics
        duration_stats = {}
        for regime, dur_list in durations.items():
            duration_stats[regime] = {
                'mean': np.mean(dur_list),
                'median': np.median(dur_list),
                'min': np.min(dur_list),
                'max': np.max(dur_list),
                'std': np.std(dur_list)
            }
        
        logger.info("Regime durations calculated")
        return duration_stats
    
    def save_model(self, filepath: str):
        """
        Save trained model to file
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save. Fit the model first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'n_regimes': self.n_regimes
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model from file
        
        Args:
            filepath: Path to load model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.n_regimes = model_data['n_regimes']
        
        logger.info(f"Model loaded from {filepath}")


class RegimeVisualizer:
    """
    Visualization tools for regime analysis
    """
    
    @staticmethod
    def plot_regimes_on_price(df: pd.DataFrame, regimes: np.ndarray, 
                              price_col: str = 'close_spot', 
                              timestamp_col: str = 'timestamp',
                              save_path: str = None):
        """
        Plot price chart with regime overlay
        
        Args:
            df: DataFrame with price data
            regimes: Array of regime labels
            price_col: Price column name
            timestamp_col: Timestamp column name
            save_path: Path to save plot
        """
        logger.info("Plotting regimes on price chart")
        
        plt.figure(figsize=(15, 7))
        
        # Define colors for regimes
        regime_colors = {-1: 'red', 0: 'gray', 1: 'green'}
        regime_names = {-1: 'Downtrend', 0: 'Sideways', 1: 'Uptrend'}
        
        # Plot price
        for regime in [-1, 0, 1]:
            mask = regimes == regime
            plt.scatter(df[timestamp_col][mask], df[price_col][mask], 
                       c=regime_colors[regime], label=regime_names[regime], 
                       alpha=0.5, s=1)
        
        plt.plot(df[timestamp_col], df[price_col], 'k-', alpha=0.3, linewidth=0.5)
        plt.xlabel('Timestamp')
        plt.ylabel('Price')
        plt.title('Market Regimes Overlay on Price Chart')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_transition_matrix(transition_matrix: np.ndarray, save_path: str = None):
        """
        Plot regime transition matrix heatmap
        
        Args:
            transition_matrix: Transition probability matrix
            save_path: Path to save plot
        """
        logger.info("Plotting transition matrix")
        
        plt.figure(figsize=(8, 6))
        
        regime_labels = ['Downtrend', 'Sideways', 'Uptrend']
        
        sns.heatmap(transition_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=regime_labels, yticklabels=regime_labels,
                   cbar_kws={'label': 'Transition Probability'})
        
        plt.xlabel('To Regime')
        plt.ylabel('From Regime')
        plt.title('Regime Transition Probability Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_regime_statistics(stats_df: pd.DataFrame, feature_cols: List[str], save_path: str = None):
        """
        Plot regime statistics
        
        Args:
            stats_df: DataFrame with regime statistics
            feature_cols: Features to plot
            save_path: Path to save plot
        """
        logger.info("Plotting regime statistics")
        
        n_features = len(feature_cols)
        fig, axes = plt.subplots(nrows=(n_features + 2) // 3, ncols=3, figsize=(15, 5 * ((n_features + 2) // 3)))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        regime_labels = {-1: 'Downtrend', 0: 'Sideways', 1: 'Uptrend'}
        
        for idx, feature in enumerate(feature_cols):
            mean_col = f'{feature}_mean'
            std_col = f'{feature}_std'
            
            if mean_col in stats_df.columns:
                x = [regime_labels.get(r, r) for r in stats_df['regime']]
                y = stats_df[mean_col]
                
                if std_col in stats_df.columns:
                    yerr = stats_df[std_col]
                    axes[idx].bar(x, y, yerr=yerr, capsize=5, alpha=0.7)
                else:
                    axes[idx].bar(x, y, alpha=0.7)
                
                axes[idx].set_title(f'{feature} by Regime')
                axes[idx].set_ylabel('Mean Value')
                axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(feature_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_duration_histogram(durations_dict: dict, save_path: str = None):
        """
        Plot regime duration histograms
        
        Args:
            durations_dict: Dictionary with durations for each regime
            save_path: Path to save plot
        """
        logger.info("Plotting duration histogram")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        regime_labels = {-1: 'Downtrend', 0: 'Sideways', 1: 'Uptrend'}
        colors = {-1: 'red', 0: 'gray', 1: 'green'}
        
        for idx, regime in enumerate([-1, 0, 1]):
            if regime in durations_dict:
                axes[idx].hist(durations_dict[regime], bins=30, alpha=0.7, color=colors[regime])
                axes[idx].set_title(f'{regime_labels[regime]} Duration')
                axes[idx].set_xlabel('Duration (periods)')
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.close()


def detect_regimes(df: pd.DataFrame, train_size: float = 0.7, 
                  feature_cols: List[str] = None) -> Tuple[pd.DataFrame, RegimeDetector]:
    """
    Complete regime detection pipeline
    
    Args:
        df: Input DataFrame
        train_size: Fraction of data to use for training
        feature_cols: Features to use for regime detection
        
    Returns:
        Tuple of (DataFrame with regimes, RegimeDetector model)
    """
    if feature_cols is None:
        feature_cols = ['avg_iv', 'iv_spread', 'pcr_oi', 'call_delta', 
                       'call_gamma', 'call_vega', 'futures_basis', 'spot_returns']
    
    logger.info("Starting regime detection pipeline")
    
    # Split data
    split_idx = int(len(df) * train_size)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # Initialize and fit detector
    detector = RegimeDetector(n_regimes=3)
    detector.fit(train_df, feature_cols)
    
    # Predict regimes
    train_regimes = detector.predict(train_df)
    test_regimes = detector.predict(test_df)
    
    # Map to semantic labels
    train_regimes_labeled = detector.map_regimes_to_labels(train_df, train_regimes)
    test_regimes_labeled = detector.map_regimes_to_labels(test_df, test_regimes)
    
    # Add to dataframe
    df_with_regimes = df.copy()
    df_with_regimes['regime'] = np.concatenate([train_regimes_labeled, test_regimes_labeled])
    
    logger.info("Regime detection completed")
    
    return df_with_regimes, detector
