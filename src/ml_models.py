"""
Machine Learning Models Module
Implement XGBoost and LSTM models for trade prediction
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import logging
import pickle

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLFeatureEngineer:
    """
    Feature engineering for ML models
    """
    
    @staticmethod
    def create_target(trades_df: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        """
        Create binary target variable (1 if trade is profitable, 0 otherwise)
        
        Args:
            trades_df: DataFrame with trade information
            df: Original DataFrame with signals
            
        Returns:
            Series with binary target
        """
        logger.info("Creating target variable")
        
        # Initialize target as NaN
        target = pd.Series(np.nan, index=df.index)
        
        # Mark profitable trades
        for _, trade in trades_df.iterrows():
            entry_idx = trade['entry_idx']
            is_profitable = 1 if trade['pnl'] > 0 else 0
            target.iloc[entry_idx] = is_profitable
        
        logger.info(f"Target created. Positive samples: {target.sum()}, Negative samples: {(target==0).sum()}")
        
        return target
    
    @staticmethod
    def add_signal_strength_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add signal strength features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with signal strength features
        """
        logger.info("Adding signal strength features")
        
        df = df.copy()
        
        # EMA gap
        if 'ema_5' in df.columns and 'ema_15' in df.columns:
            df['ema_gap'] = df['ema_5'] - df['ema_15']
            df['ema_gap_pct'] = (df['ema_gap'] / df['ema_15']) * 100
        
        # ATR (Average True Range)
        if 'high' in df.columns and 'low' in df.columns:
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close_spot'].shift(1)),
                    abs(df['low'] - df['close_spot'].shift(1))
                )
            )
            df['atr_14'] = df['tr'].rolling(window=14).mean()
        
        # Volume features
        if 'volume_spot' in df.columns:
            df['volume_ma_20'] = df['volume_spot'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume_spot'] / (df['volume_ma_20'] + 1e-10)
        
        # Momentum
        if 'close_spot' in df.columns:
            df['momentum_5'] = df['close_spot'] - df['close_spot'].shift(5)
            df['momentum_10'] = df['close_spot'] - df['close_spot'].shift(10)
            df['roc_5'] = df['close_spot'].pct_change(5) * 100
        
        logger.info("Signal strength features added")
        return df
    
    @staticmethod
    def prepare_ml_features(df: pd.DataFrame, feature_cols: List[str] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for ML models
        
        Args:
            df: Input DataFrame
            feature_cols: List of feature columns (if None, auto-select)
            
        Returns:
            Tuple of (DataFrame, feature_columns)
        """
        logger.info("Preparing ML features")
        
        if feature_cols is None:
            # Auto-select features
            feature_cols = [
                'ema_5', 'ema_15', 'ema_gap', 'ema_gap_pct',
                'avg_iv', 'iv_spread', 'pcr_oi', 'pcr_volume',
                'call_delta', 'call_gamma', 'call_vega', 'call_theta',
                'put_delta', 'put_gamma', 'put_vega', 'put_theta',
                'futures_basis', 'spot_returns', 'futures_returns',
                'delta_neutral_ratio', 'gamma_exposure',
                'regime', 'hour', 'day_of_week',
                'atr_14', 'volume_ratio', 'momentum_5', 'roc_5'
            ]
            
            # Filter to existing columns
            feature_cols = [col for col in feature_cols if col in df.columns]
        
        logger.info(f"Selected {len(feature_cols)} features for ML")
        
        return df, feature_cols


class XGBoostModel:
    """
    XGBoost classifier for trade prediction
    """
    
    def __init__(self, params: Dict = None):
        """
        Initialize XGBoost model
        
        Args:
            params: XGBoost parameters
        """
        self.params = params or {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        self.model = None
        self.feature_cols = None
        self.scaler = StandardScaler()
        
        logger.info("Initialized XGBoost model")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
             X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
        """
        logger.info("Training XGBoost model")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize model
        self.model = xgb.XGBClassifier(**self.params)
        
        # Train with early stopping if validation set provided
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train_scaled, y_train, verbose=False)
        
        # Calculate training metrics
        y_train_pred = self.model.predict(X_train_scaled)
        train_acc = np.mean(y_train_pred == y_train)
        
        logger.info(f"XGBoost training completed. Training accuracy: {train_acc:.4f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Features
            
        Returns:
            Predicted class labels
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Features
            
        Returns:
            Predicted probabilities
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str):
        """
        Save model to file
        
        Args:
            filepath: Path to save model
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'params': self.params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"XGBoost model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model from file
        
        Args:
            filepath: Path to load model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.params = model_data['params']
        
        logger.info(f"XGBoost model loaded from {filepath}")


class LSTMModel:
    """
    LSTM model for sequential trade prediction
    """
    
    def __init__(self, sequence_length: int = 10, n_features: int = 20, 
                 lstm_units: int = 64, dropout: float = 0.2):
        """
        Initialize LSTM model
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of features
            lstm_units: Number of LSTM units
            dropout: Dropout rate
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.model = None
        self.scaler = StandardScaler()
        
        logger.info(f"Initialized LSTM model (seq_len={sequence_length}, features={n_features})")
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Tuple:
        """
        Create sequences for LSTM
        
        Args:
            X: Features array
            y: Target array
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_seq = []
        y_seq = [] if y is not None else None
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            if y is not None:
                y_seq.append(y[i + self.sequence_length])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq) if y is not None else None
        
        return X_seq, y_seq
    
    def build_model(self):
        """
        Build LSTM architecture
        """
        from tensorflow import keras
        from tensorflow.keras import layers
        
        logger.info("Building LSTM model architecture")
        
        model = keras.Sequential([
            layers.LSTM(self.lstm_units, input_shape=(self.sequence_length, self.n_features), 
                       return_sequences=False),
            layers.Dropout(self.dropout),
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.dropout),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        self.model = model
        logger.info("LSTM model built")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
             X_val: pd.DataFrame = None, y_val: pd.Series = None,
             epochs: int = 50, batch_size: int = 32):
        """
        Train LSTM model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            epochs: Number of epochs
            batch_size: Batch size
        """
        from tensorflow.keras.callbacks import EarlyStopping
        
        logger.info("Training LSTM model")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train.values)
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_seq, y_val_seq = self.create_sequences(X_val_scaled, y_val.values)
            validation_data = (X_val_seq, y_val_seq)
        
        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss' if validation_data else 'loss',
                                  patience=10, restore_best_weights=True)
        
        # Train
        history = self.model.fit(
            X_train_seq, y_train_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=[early_stop],
            verbose=0
        )
        
        logger.info(f"LSTM training completed. Final loss: {history.history['loss'][-1]:.4f}")
        
        return history
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Features
            
        Returns:
            Predicted class labels
        """
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self.create_sequences(X_scaled)
        
        predictions = self.model.predict(X_seq, verbose=0)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Features
            
        Returns:
            Predicted probabilities
        """
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self.create_sequences(X_scaled)
        
        predictions = self.model.predict(X_seq, verbose=0)
        
        # Pad with NaN for sequence length
        full_predictions = np.full(len(X), np.nan)
        full_predictions[self.sequence_length:] = predictions.flatten()
        
        return full_predictions
    
    def save_model(self, filepath: str):
        """
        Save model to file
        
        Args:
            filepath: Path to save model
        """
        self.model.save(filepath)
        
        # Save scaler separately
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"LSTM model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model from file
        
        Args:
            filepath: Path to load model from
        """
        from tensorflow import keras
        
        self.model = keras.models.load_model(filepath)
        
        # Load scaler
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        logger.info(f"LSTM model loaded from {filepath}")


class ModelEvaluator:
    """
    Evaluate ML models
    """
    
    @staticmethod
    def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, 
                      y_pred_proba: np.ndarray = None) -> Dict:
        """
        Evaluate model performance
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model performance")
        
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = np.mean(y_true == y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics['precision'] = report['1']['precision']
        metrics['recall'] = report['1']['recall']
        metrics['f1_score'] = report['1']['f1-score']
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # AUC if probabilities provided
        if y_pred_proba is not None:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        
        logger.info(f"Model evaluation: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
        
        return metrics
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, save_path: str = None):
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save plot
        """
        import matplotlib.pyplot as plt
        
        logger.info("Plotting ROC curve")
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.close()


def train_ml_models(df: pd.DataFrame, feature_cols: List[str], target_col: str,
                   train_size: float = 0.7) -> Tuple[XGBoostModel, LSTMModel]:
    """
    Train both XGBoost and LSTM models
    
    Args:
        df: Input DataFrame with features and target
        feature_cols: List of feature columns
        target_col: Target column name
        train_size: Training set size fraction
        
    Returns:
        Tuple of (XGBoost model, LSTM model)
    """
    logger.info("Training ML models")
    
    # Remove rows with missing target
    df_clean = df.dropna(subset=[target_col]).copy()
    
    # Split data
    split_idx = int(len(df_clean) * train_size)
    
    X_train = df_clean[feature_cols].iloc[:split_idx]
    y_train = df_clean[target_col].iloc[:split_idx]
    X_test = df_clean[feature_cols].iloc[split_idx:]
    y_test = df_clean[target_col].iloc[split_idx:]
    
    # Handle missing values in features
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Train XGBoost
    logger.info("Training XGBoost...")
    xgb_model = XGBoostModel()
    xgb_model.train(X_train, y_train, X_test, y_test)
    
    # Train LSTM
    logger.info("Training LSTM...")
    lstm_model = LSTMModel(sequence_length=10, n_features=len(feature_cols))
    lstm_model.train(X_train, y_train, X_test, y_test, epochs=50)
    
    logger.info("ML models training completed")
    
    return xgb_model, lstm_model
