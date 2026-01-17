"""
Greeks Calculation Module
Calculate Black-Scholes Greeks for options
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlackScholesGreeks:
    """
    Calculate Greeks using Black-Scholes model
    """
    
    def __init__(self, risk_free_rate: float = 0.065):
        """
        Initialize Greeks calculator
        
        Args:
            risk_free_rate: Risk-free rate (default: 6.5% = 0.065)
        """
        self.risk_free_rate = risk_free_rate
        logger.info(f"Initialized Black-Scholes Greeks calculator with r={risk_free_rate}")
    
    @staticmethod
    def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate d1 parameter
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (in years)
            r: Risk-free rate
            sigma: Implied volatility
            
        Returns:
            d1 value
        """
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate d2 parameter
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (in years)
            r: Risk-free rate
            sigma: Implied volatility
            
        Returns:
            d2 value
        """
        d1 = BlackScholesGreeks._d1(S, K, T, r, sigma)
        return d1 - sigma * np.sqrt(T)
    
    def calculate_delta(self, S: float, K: float, T: float, sigma: float, option_type: str = 'call') -> float:
        """
        Calculate Delta
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (in years)
            sigma: Implied volatility
            option_type: 'call' or 'put'
            
        Returns:
            Delta value
        """
        d1 = self._d1(S, K, T, self.risk_free_rate, sigma)
        
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
        else:  # put
            delta = norm.cdf(d1) - 1
        
        return delta
    
    def calculate_gamma(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate Gamma (same for both call and put)
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (in years)
            sigma: Implied volatility
            
        Returns:
            Gamma value
        """
        d1 = self._d1(S, K, T, self.risk_free_rate, sigma)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        return gamma
    
    def calculate_theta(self, S: float, K: float, T: float, sigma: float, option_type: str = 'call') -> float:
        """
        Calculate Theta
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (in years)
            sigma: Implied volatility
            option_type: 'call' or 'put'
            
        Returns:
            Theta value (per day)
        """
        d1 = self._d1(S, K, T, self.risk_free_rate, sigma)
        d2 = self._d2(S, K, T, self.risk_free_rate, sigma)
        
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        
        if option_type.lower() == 'call':
            term2 = -self.risk_free_rate * K * np.exp(-self.risk_free_rate * T) * norm.cdf(d2)
            theta = term1 + term2
        else:  # put
            term2 = self.risk_free_rate * K * np.exp(-self.risk_free_rate * T) * norm.cdf(-d2)
            theta = term1 + term2
        
        # Convert to per-day theta
        return theta / 365
    
    def calculate_vega(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate Vega (same for both call and put)
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (in years)
            sigma: Implied volatility
            
        Returns:
            Vega value (per 1% change in IV)
        """
        d1 = self._d1(S, K, T, self.risk_free_rate, sigma)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        # Convert to per 1% change
        return vega / 100
    
    def calculate_rho(self, S: float, K: float, T: float, sigma: float, option_type: str = 'call') -> float:
        """
        Calculate Rho
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (in years)
            sigma: Implied volatility
            option_type: 'call' or 'put'
            
        Returns:
            Rho value (per 1% change in interest rate)
        """
        d2 = self._d2(S, K, T, self.risk_free_rate, sigma)
        
        if option_type.lower() == 'call':
            rho = K * T * np.exp(-self.risk_free_rate * T) * norm.cdf(d2)
        else:  # put
            rho = -K * T * np.exp(-self.risk_free_rate * T) * norm.cdf(-d2)
        
        # Convert to per 1% change
        return rho / 100
    
    def calculate_all_greeks(self, S: float, K: float, T: float, sigma: float, option_type: str = 'call') -> dict:
        """
        Calculate all Greeks
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (in years)
            sigma: Implied volatility
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary with all Greeks
        """
        greeks = {
            'delta': self.calculate_delta(S, K, T, sigma, option_type),
            'gamma': self.calculate_gamma(S, K, T, sigma),
            'theta': self.calculate_theta(S, K, T, sigma, option_type),
            'vega': self.calculate_vega(S, K, T, sigma),
            'rho': self.calculate_rho(S, K, T, sigma, option_type)
        }
        
        return greeks


class GreeksCalculator:
    """
    High-level Greeks calculator for DataFrames
    """
    
    def __init__(self, risk_free_rate: float = 0.065):
        """
        Initialize Greeks calculator
        
        Args:
            risk_free_rate: Risk-free rate (default: 6.5%)
        """
        self.bs = BlackScholesGreeks(risk_free_rate)
    
    def calculate_time_to_expiry(self, current_date, expiry_date) -> float:
        """
        Calculate time to expiry in years
        
        Args:
            current_date: Current date
            expiry_date: Option expiry date
            
        Returns:
            Time to expiry in years
        """
        import pandas as pd
        
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
        if isinstance(expiry_date, str):
            expiry_date = pd.to_datetime(expiry_date)
        
        days_to_expiry = (expiry_date - current_date).days
        return max(days_to_expiry / 365.0, 1/365)  # Minimum 1 day
    
    def add_greeks_to_dataframe(self, df, spot_col='close_spot', strike_col='strike', 
                               iv_col='iv', time_to_expiry=None, option_type='call'):
        """
        Add Greeks columns to DataFrame
        
        Args:
            df: Input DataFrame
            spot_col: Spot price column name
            strike_col: Strike price column name
            iv_col: Implied volatility column name
            time_to_expiry: Time to expiry in years (if None, will be calculated)
            option_type: 'call' or 'put'
            
        Returns:
            DataFrame with Greeks columns added
        """
        import pandas as pd
        
        logger.info(f"Calculating Greeks for {option_type} options")
        
        df = df.copy()
        
        # Calculate Greeks for each row
        deltas = []
        gammas = []
        thetas = []
        vegas = []
        rhos = []
        
        for idx, row in df.iterrows():
            S = row[spot_col]
            K = row[strike_col]
            sigma = row[iv_col]
            
            # Calculate or use provided time to expiry
            if time_to_expiry is None:
                if 'expiry_date' in df.columns and 'timestamp' in df.columns:
                    T = self.calculate_time_to_expiry(row['timestamp'], row['expiry_date'])
                else:
                    T = 30 / 365  # Default to 30 days
            else:
                T = time_to_expiry
            
            # Handle edge cases
            if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
                deltas.append(np.nan)
                gammas.append(np.nan)
                thetas.append(np.nan)
                vegas.append(np.nan)
                rhos.append(np.nan)
                continue
            
            greeks = self.bs.calculate_all_greeks(S, K, T, sigma, option_type)
            
            deltas.append(greeks['delta'])
            gammas.append(greeks['gamma'])
            thetas.append(greeks['theta'])
            vegas.append(greeks['vega'])
            rhos.append(greeks['rho'])
        
        # Add Greeks columns
        prefix = f'{option_type}_'
        df[f'{prefix}delta'] = deltas
        df[f'{prefix}gamma'] = gammas
        df[f'{prefix}theta'] = thetas
        df[f'{prefix}vega'] = vegas
        df[f'{prefix}rho'] = rhos
        
        logger.info(f"Greeks calculated for {len(df)} rows")
        return df


def calculate_implied_volatility(option_price: float, S: float, K: float, T: float, 
                                 r: float = 0.065, option_type: str = 'call', 
                                 max_iterations: int = 100, tolerance: float = 1e-5) -> float:
    """
    Calculate implied volatility using Newton-Raphson method
    
    Args:
        option_price: Market price of option
        S: Spot price
        K: Strike price
        T: Time to expiry (in years)
        r: Risk-free rate
        option_type: 'call' or 'put'
        max_iterations: Maximum iterations for convergence
        tolerance: Tolerance for convergence
        
    Returns:
        Implied volatility
    """
    from scipy.optimize import newton
    
    def black_scholes_price(sigma):
        """Calculate Black-Scholes price"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price - option_price
    
    try:
        # Initial guess
        iv = 0.3
        iv = newton(black_scholes_price, iv, maxiter=max_iterations, tol=tolerance)
        return max(iv, 0.01)  # Ensure positive IV
    except:
        return np.nan


def calculate_atm_greeks(df, spot_col='close_spot', strikes_range=2):
    """
    Calculate Greeks for ATM options (ATM ± strikes_range)
    
    Args:
        df: Input DataFrame
        spot_col: Spot price column
        strikes_range: Number of strikes above and below ATM
        
    Returns:
        DataFrame with ATM Greeks
    """
    logger.info(f"Calculating ATM Greeks for ±{strikes_range} strikes")
    
    calculator = GreeksCalculator()
    
    # Calculate Greeks for calls and puts at different strikes
    # This would be implemented based on the specific data structure
    
    return df
