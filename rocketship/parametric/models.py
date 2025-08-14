"""
Parametric fitting models for ROCKETSHIP.

T1, T2, T2*, ADC and other parametric map calculation.
"""

import numpy as np
from scipy import optimize
from typing import Tuple, Optional, Dict, Any, List
import logging


logger = logging.getLogger(__name__)


class ParametricModel:
    """Base class for parametric fitting models."""
    
    def __init__(self, name: str):
        self.name = name
        self.parameter_names = []
        self.bounds = {}
        self.initial_values = {}
    
    def model_function(self, x: np.ndarray, *params) -> np.ndarray:
        """Model function - must be implemented by subclasses."""
        raise NotImplementedError
    
    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Fit model to data."""
        raise NotImplementedError


class T1Model(ParametricModel):
    """T1 relaxation time fitting models."""
    
    def __init__(self, fit_type: str = 'inversion_recovery'):
        super().__init__(f"T1_{fit_type}")
        self.fit_type = fit_type
        
        if fit_type == 'inversion_recovery':
            # S(TI) = S0 * (1 - 2*exp(-TI/T1))
            self.parameter_names = ['S0', 'T1']
            self.bounds = {'S0': (0, np.inf), 'T1': (1, 10000)}
            self.initial_values = {'S0': 1000, 'T1': 1000}
            
        elif fit_type == 'saturation_recovery':
            # S(TR) = S0 * (1 - exp(-TR/T1))
            self.parameter_names = ['S0', 'T1']
            self.bounds = {'S0': (0, np.inf), 'T1': (1, 10000)}
            self.initial_values = {'S0': 1000, 'T1': 1000}
            
        elif fit_type == 'variable_flip_angle':
            # S(α) = S0 * sin(α) * (1-exp(-TR/T1)) / (1-cos(α)*exp(-TR/T1))
            self.parameter_names = ['S0', 'T1']
            self.bounds = {'S0': (0, np.inf), 'T1': (1, 10000)}
            self.initial_values = {'S0': 1000, 'T1': 1000}
    
    def model_function(self, x: np.ndarray, s0: float, t1: float, tr: Optional[float] = None) -> np.ndarray:
        """T1 model functions."""
        if self.fit_type == 'inversion_recovery':
            # x = TI (inversion times)
            return np.abs(s0 * (1 - 2 * np.exp(-x / t1)))
            
        elif self.fit_type == 'saturation_recovery':
            # x = TR (repetition times)
            return s0 * (1 - np.exp(-x / t1))
            
        elif self.fit_type == 'variable_flip_angle':
            # x = flip angles in degrees, tr must be provided
            if tr is None:
                raise ValueError("TR must be provided for variable flip angle fitting")
            alpha_rad = np.deg2rad(x)
            return s0 * np.sin(alpha_rad) * (1 - np.exp(-tr / t1)) / (1 - np.cos(alpha_rad) * np.exp(-tr / t1))
        
        else:
            raise ValueError(f"Unknown T1 fit type: {self.fit_type}")
    
    def fit(self, x: np.ndarray, y: np.ndarray, tr: Optional[float] = None, 
            weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Fit T1 model to data."""
        
        # Remove invalid data points
        valid_mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
        if not np.any(valid_mask):
            return {'S0': 0, 'T1': 0, 'r_squared': 0, 'success': False}
        
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        if weights is not None:
            weights = weights[valid_mask]
        
        # Initial parameter estimation
        s0_init = np.max(y_valid) * 1.2
        
        if self.fit_type == 'inversion_recovery':
            # Find null point for better T1 estimate
            min_idx = np.argmin(np.abs(y_valid))
            if min_idx > 0:
                ti_null = x_valid[min_idx]
                t1_init = ti_null / np.log(2)  # Approximate relationship
            else:
                t1_init = 1000
        else:
            t1_init = 1000
        
        # Define objective function
        def objective(params):
            s0, t1 = params
            if t1 <= 0 or s0 <= 0:
                return np.inf
            
            try:
                model_y = self.model_function(x_valid, s0, t1, tr)
                residuals = y_valid - model_y
                
                if weights is not None:
                    residuals *= weights
                
                return np.sum(residuals**2)
            except:
                return np.inf
        
        # Optimization bounds
        bounds = [
            (self.bounds['S0'][0], self.bounds['S0'][1]),
            (self.bounds['T1'][0], self.bounds['T1'][1])
        ]
        
        try:
            result = optimize.minimize(
                objective,
                [s0_init, t1_init],
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if result.success and result.fun < np.inf:
                s0_opt, t1_opt = result.x
                model_y = self.model_function(x_valid, s0_opt, t1_opt, tr)
                
                # Calculate R-squared
                ss_res = np.sum((y_valid - model_y)**2)
                ss_tot = np.sum((y_valid - np.mean(y_valid))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                return {
                    'S0': s0_opt,
                    'T1': t1_opt,
                    'r_squared': max(0, r_squared),
                    'residual_norm': result.fun,
                    'success': True,
                    'model_fit': model_y
                }
            else:
                return {'S0': 0, 'T1': 0, 'r_squared': 0, 'success': False}
                
        except Exception as e:
            logger.warning(f"T1 fitting failed: {e}")
            return {'S0': 0, 'T1': 0, 'r_squared': 0, 'success': False}


class T2Model(ParametricModel):
    """T2 relaxation time fitting model."""
    
    def __init__(self):
        super().__init__("T2")
        self.parameter_names = ['S0', 'T2']
        self.bounds = {'S0': (0, np.inf), 'T2': (1, 1000)}
        self.initial_values = {'S0': 1000, 'T2': 100}
    
    def model_function(self, te: np.ndarray, s0: float, t2: float) -> np.ndarray:
        """T2 decay model: S(TE) = S0 * exp(-TE/T2)."""
        return s0 * np.exp(-te / t2)
    
    def fit(self, te: np.ndarray, signal: np.ndarray, 
            linear_fit: bool = False) -> Dict[str, Any]:
        """Fit T2 model to data."""
        
        # Remove invalid data points
        valid_mask = np.isfinite(te) & np.isfinite(signal) & (signal > 0)
        if not np.any(valid_mask):
            return {'S0': 0, 'T2': 0, 'r_squared': 0, 'success': False}
        
        te_valid = te[valid_mask]
        signal_valid = signal[valid_mask]
        
        if linear_fit:
            # Linear fitting in log space: ln(S) = ln(S0) - TE/T2
            try:
                log_signal = np.log(signal_valid)
                
                # Linear regression
                A = np.vstack([np.ones(len(te_valid)), te_valid]).T
                coeffs, residuals, rank, s = np.linalg.lstsq(A, log_signal, rcond=None)
                
                s0_opt = np.exp(coeffs[0])
                t2_opt = -1.0 / coeffs[1] if coeffs[1] < 0 else 1000
                
                # Calculate R-squared
                model_signal = self.model_function(te_valid, s0_opt, t2_opt)
                ss_res = np.sum((signal_valid - model_signal)**2)
                ss_tot = np.sum((signal_valid - np.mean(signal_valid))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                return {
                    'S0': s0_opt,
                    'T2': max(1, t2_opt),  # Ensure positive T2
                    'r_squared': max(0, r_squared),
                    'success': True
                }
                
            except Exception as e:
                logger.warning(f"Linear T2 fitting failed: {e}")
                return {'S0': 0, 'T2': 0, 'r_squared': 0, 'success': False}
        
        else:
            # Nonlinear fitting
            s0_init = signal_valid[0] if len(signal_valid) > 0 else 1000
            t2_init = 100
            
            def objective(params):
                s0, t2 = params
                if t2 <= 0 or s0 <= 0:
                    return np.inf
                
                model_signal = self.model_function(te_valid, s0, t2)
                return np.sum((signal_valid - model_signal)**2)
            
            bounds = [
                (self.bounds['S0'][0], self.bounds['S0'][1]),
                (self.bounds['T2'][0], self.bounds['T2'][1])
            ]
            
            try:
                result = optimize.minimize(
                    objective,
                    [s0_init, t2_init],
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                
                if result.success:
                    s0_opt, t2_opt = result.x
                    model_signal = self.model_function(te_valid, s0_opt, t2_opt)
                    
                    # Calculate R-squared
                    ss_res = np.sum((signal_valid - model_signal)**2)
                    ss_tot = np.sum((signal_valid - np.mean(signal_valid))**2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    return {
                        'S0': s0_opt,
                        'T2': t2_opt,
                        'r_squared': max(0, r_squared),
                        'residual_norm': result.fun,
                        'success': True
                    }
                else:
                    return {'S0': 0, 'T2': 0, 'r_squared': 0, 'success': False}
                    
            except Exception as e:
                logger.warning(f"Nonlinear T2 fitting failed: {e}")
                return {'S0': 0, 'T2': 0, 'r_squared': 0, 'success': False}


class T2StarModel(T2Model):
    """T2* relaxation time fitting model (same as T2 but typically shorter values)."""
    
    def __init__(self):
        super().__init__()
        self.name = "T2*"
        self.bounds = {'S0': (0, np.inf), 'T2': (1, 200)}  # Shorter range for T2*


class ADCModel(ParametricModel):
    """Apparent Diffusion Coefficient (ADC) fitting model."""
    
    def __init__(self):
        super().__init__("ADC")
        self.parameter_names = ['S0', 'ADC']
        self.bounds = {'S0': (0, np.inf), 'ADC': (0, 0.005)}  # ADC in mm²/s
        self.initial_values = {'S0': 1000, 'ADC': 0.001}
    
    def model_function(self, b: np.ndarray, s0: float, adc: float) -> np.ndarray:
        """ADC model: S(b) = S0 * exp(-b*ADC)."""
        return s0 * np.exp(-b * adc)
    
    def fit(self, b_values: np.ndarray, signal: np.ndarray, 
            linear_fit: bool = True) -> Dict[str, Any]:
        """Fit ADC model to diffusion data."""
        
        # Remove invalid data points
        valid_mask = np.isfinite(b_values) & np.isfinite(signal) & (signal > 0)
        if not np.any(valid_mask):
            return {'S0': 0, 'ADC': 0, 'r_squared': 0, 'success': False}
        
        b_valid = b_values[valid_mask]
        signal_valid = signal[valid_mask]
        
        if linear_fit:
            # Linear fitting in log space: ln(S) = ln(S0) - b*ADC
            try:
                log_signal = np.log(signal_valid)
                
                # Linear regression
                A = np.vstack([np.ones(len(b_valid)), b_valid]).T
                coeffs, residuals, rank, s = np.linalg.lstsq(A, log_signal, rcond=None)
                
                s0_opt = np.exp(coeffs[0])
                adc_opt = -coeffs[1] if coeffs[1] < 0 else 0
                
                # Calculate R-squared in linear space
                model_signal = self.model_function(b_valid, s0_opt, adc_opt)
                ss_res = np.sum((signal_valid - model_signal)**2)
                ss_tot = np.sum((signal_valid - np.mean(signal_valid))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                return {
                    'S0': s0_opt,
                    'ADC': max(0, adc_opt),  # Ensure non-negative ADC
                    'r_squared': max(0, r_squared),
                    'success': True
                }
                
            except Exception as e:
                logger.warning(f"Linear ADC fitting failed: {e}")
                return {'S0': 0, 'ADC': 0, 'r_squared': 0, 'success': False}
        
        else:
            # Nonlinear fitting (similar to T2 fitting)
            s0_init = signal_valid[0] if len(signal_valid) > 0 else 1000
            adc_init = 0.001
            
            def objective(params):
                s0, adc = params
                if adc < 0 or s0 <= 0:
                    return np.inf
                
                model_signal = self.model_function(b_valid, s0, adc)
                return np.sum((signal_valid - model_signal)**2)
            
            bounds = [
                (self.bounds['S0'][0], self.bounds['S0'][1]),
                (self.bounds['ADC'][0], self.bounds['ADC'][1])
            ]
            
            try:
                result = optimize.minimize(
                    objective,
                    [s0_init, adc_init],
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                
                if result.success:
                    s0_opt, adc_opt = result.x
                    model_signal = self.model_function(b_valid, s0_opt, adc_opt)
                    
                    # Calculate R-squared
                    ss_res = np.sum((signal_valid - model_signal)**2)
                    ss_tot = np.sum((signal_valid - np.mean(signal_valid))**2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    return {
                        'S0': s0_opt,
                        'ADC': adc_opt,
                        'r_squared': max(0, r_squared),
                        'residual_norm': result.fun,
                        'success': True
                    }
                else:
                    return {'S0': 0, 'ADC': 0, 'r_squared': 0, 'success': False}
                    
            except Exception as e:
                logger.warning(f"Nonlinear ADC fitting failed: {e}")
                return {'S0': 0, 'ADC': 0, 'r_squared': 0, 'success': False}