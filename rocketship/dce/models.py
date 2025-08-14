"""
DCE-MRI models for ROCKETSHIP.

Pharmacokinetic models for DCE-MRI analysis including Tofts, Extended Tofts, Patlak, etc.
"""

import numpy as np
from scipy import optimize, integrate, interpolate
from typing import Tuple, Optional, Dict, Any, Callable
import logging


logger = logging.getLogger(__name__)


class DCEModel:
    """Base class for DCE-MRI pharmacokinetic models."""
    
    def __init__(self, name: str):
        self.name = name
        self.parameter_names = []
        self.parameter_bounds = {}
        self.initial_values = {}
    
    def model_function(self, t: np.ndarray, *params) -> np.ndarray:
        """Model function - must be implemented by subclasses."""
        raise NotImplementedError
    
    def fit(self, t: np.ndarray, ct: np.ndarray, cp: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Fit model to data."""
        raise NotImplementedError


class ToftsModel(DCEModel):
    """
    Standard Tofts model for DCE-MRI.
    
    Ct(t) = Ktrans * Cp(t) ⊗ exp(-Ktrans*t/ve)
    
    Parameters:
        - Ktrans: volume transfer constant
        - ve: extravascular extracellular space volume fraction
    """
    
    def __init__(self):
        super().__init__("Tofts")
        self.parameter_names = ['Ktrans', 've']
        self.parameter_bounds = {
            'Ktrans': (1e-7, 2.0),
            've': (0.02, 1.0)
        }
        self.initial_values = {
            'Ktrans': 0.0002,
            've': 0.2
        }
    
    def model_function(self, t: np.ndarray, ktrans: float, ve: float, cp: np.ndarray) -> np.ndarray:
        """
        Tofts model function.
        
        Args:
            t: Time points
            ktrans: Volume transfer constant
            ve: Extravascular extracellular volume fraction
            cp: Arterial input function
            
        Returns:
            Modeled concentration
        """
        if ktrans <= 0 or ve <= 0:
            return np.zeros_like(t)
        
        # Convolution with exponential decay
        kep = ktrans / ve
        exp_decay = np.exp(-kep * t)
        
        # Convolution using numerical integration
        ct = np.zeros_like(t)
        dt = np.mean(np.diff(t)) if len(t) > 1 else 1.0
        
        for i in range(len(t)):
            if i == 0:
                ct[i] = 0
            else:
                # Trapezoidal integration
                integrand = cp[:i+1] * np.exp(-kep * (t[i] - t[:i+1]))
                ct[i] = ktrans * np.trapz(integrand, t[:i+1])
        
        return ct
    
    def fit(self, t: np.ndarray, ct: np.ndarray, cp: np.ndarray, 
            bounds: Optional[Dict[str, Tuple[float, float]]] = None,
            initial: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Fit Tofts model to data.
        
        Args:
            t: Time points
            ct: Tissue concentration
            cp: Arterial input function
            bounds: Parameter bounds (optional)
            initial: Initial parameter values (optional)
            
        Returns:
            Fitting results dictionary
        """
        # Set bounds and initial values
        if bounds is None:
            bounds = self.parameter_bounds
        if initial is None:
            initial = self.initial_values
        
        # Define objective function
        def objective(params):
            ktrans, ve = params
            model_ct = self.model_function(t, ktrans, ve, cp)
            residuals = ct - model_ct
            return np.sum(residuals**2)
        
        # Set up bounds for optimization
        param_bounds = [(bounds['Ktrans'], bounds['ve'])]
        initial_guess = [initial['Ktrans'], initial['ve']]
        
        try:
            # Use scipy optimization
            result = optimize.minimize(
                objective, 
                initial_guess, 
                bounds=param_bounds,
                method='L-BFGS-B'
            )
            
            if result.success:
                ktrans_opt, ve_opt = result.x
                model_ct = self.model_function(t, ktrans_opt, ve_opt, cp)
                
                # Calculate R-squared
                ss_res = np.sum((ct - model_ct)**2)
                ss_tot = np.sum((ct - np.mean(ct))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                return {
                    'Ktrans': ktrans_opt,
                    've': ve_opt,
                    'r_squared': r_squared,
                    'residual_norm': result.fun,
                    'success': True,
                    'model_fit': model_ct
                }
            else:
                logger.warning(f"Optimization failed: {result.message}")
                return {
                    'Ktrans': 0,
                    've': 0,
                    'r_squared': 0,
                    'residual_norm': np.inf,
                    'success': False,
                    'model_fit': np.zeros_like(t)
                }
                
        except Exception as e:
            logger.error(f"Error in Tofts fitting: {e}")
            return {
                'Ktrans': 0,
                've': 0,
                'r_squared': 0,
                'residual_norm': np.inf,
                'success': False,
                'model_fit': np.zeros_like(t)
            }


class ExtendedToftsModel(DCEModel):
    """
    Extended Tofts model including vascular component.
    
    Ct(t) = vp*Cp(t) + Ktrans * Cp(t) ⊗ exp(-Ktrans*t/ve)
    
    Parameters:
        - Ktrans: volume transfer constant
        - ve: extravascular extracellular space volume fraction
        - vp: plasma volume fraction
    """
    
    def __init__(self):
        super().__init__("Extended Tofts")
        self.parameter_names = ['Ktrans', 've', 'vp']
        self.parameter_bounds = {
            'Ktrans': (1e-7, 2.0),
            've': (0.02, 1.0),
            'vp': (0.001, 1.0)
        }
        self.initial_values = {
            'Ktrans': 0.0002,
            've': 0.2,
            'vp': 0.02
        }
    
    def model_function(self, t: np.ndarray, ktrans: float, ve: float, vp: float, cp: np.ndarray) -> np.ndarray:
        """Extended Tofts model function."""
        if ktrans <= 0 or ve <= 0 or vp < 0:
            return np.zeros_like(t)
        
        # Vascular component
        vascular_term = vp * cp
        
        # Extravascular component (same as standard Tofts)
        tofts_model = ToftsModel()
        extravascular_term = tofts_model.model_function(t, ktrans, ve, cp)
        
        return vascular_term + extravascular_term


class PatlakModel(DCEModel):
    """
    Patlak model for DCE-MRI.
    
    Ct(t)/Cp(t) = Ktrans * ∫Cp(τ)dτ/Cp(t) + vp
    
    Parameters:
        - Ktrans: volume transfer constant
        - vp: plasma volume fraction
    """
    
    def __init__(self):
        super().__init__("Patlak")
        self.parameter_names = ['Ktrans', 'vp']
        self.parameter_bounds = {
            'Ktrans': (1e-7, 2.0),
            'vp': (0.001, 1.0)
        }
        self.initial_values = {
            'Ktrans': 0.0002,
            'vp': 0.02
        }
    
    def model_function(self, t: np.ndarray, ktrans: float, vp: float, cp: np.ndarray) -> np.ndarray:
        """Patlak model function."""
        if ktrans <= 0 or vp < 0:
            return np.zeros_like(t)
        
        ct = np.zeros_like(t)
        
        for i in range(len(t)):
            if i == 0:
                ct[i] = vp * cp[i]
            else:
                # Calculate integral of Cp
                integral_cp = np.trapz(cp[:i+1], t[:i+1])
                ct[i] = ktrans * integral_cp + vp * cp[i]
        
        return ct


class AIFModel:
    """Arterial Input Function models."""
    
    @staticmethod
    def biexponential_aif(t: np.ndarray, A: float, B: float, c: float, d: float, 
                         baseline: float = 0.0) -> np.ndarray:
        """
        Biexponential AIF model.
        
        Cp(t) = A*exp(-c*t) + B*exp(-d*t) + baseline
        
        Args:
            t: Time points
            A, B: Amplitude parameters
            c, d: Decay constants
            baseline: Baseline value
            
        Returns:
            AIF values
        """
        return A * np.exp(-c * t) + B * np.exp(-d * t) + baseline
    
    @staticmethod
    def parker_aif(t: np.ndarray, 
                   A1: float = 0.809, A2: float = 0.330,
                   T1: float = 0.17047, T2: float = 0.365,
                   sigma1: float = 0.0563, sigma2: float = 0.132,
                   alpha: float = 1.050, beta: float = 0.1685,
                   s: float = 38.078, tau: float = 0.483) -> np.ndarray:
        """
        Parker population AIF model.
        
        Args:
            t: Time points (in minutes)
            Various model parameters with default values from Parker et al.
            
        Returns:
            AIF values
        """
        # Convert time from seconds to minutes if needed
        if np.max(t) > 10:  # Assume seconds if max > 10
            t_min = t / 60.0
        else:
            t_min = t
        
        # Two Gaussian functions plus exponential decay
        gaussian1 = A1 / (sigma1 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((t_min - T1) / sigma1)**2)
        gaussian2 = A2 / (sigma2 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((t_min - T2) / sigma2)**2)
        exponential = alpha * np.exp(-beta * t_min) / (1 + np.exp(-s * (t_min - tau)))
        
        return gaussian1 + gaussian2 + exponential
    
    @staticmethod
    def fit_biexponential_aif(t: np.ndarray, cp: np.ndarray,
                             bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Any]:
        """
        Fit biexponential model to AIF data.
        
        Args:
            t: Time points
            cp: AIF concentration values
            bounds: Parameter bounds
            
        Returns:
            Fitting results
        """
        if bounds is None:
            bounds = {
                'A': (0, 5),
                'B': (0, 5),
                'c': (0, 50),
                'd': (0, 50)
            }
        
        def objective(params):
            A, B, c, d = params
            model_cp = AIFModel.biexponential_aif(t, A, B, c, d)
            return np.sum((cp - model_cp)**2)
        
        # Initial guess
        A_init = np.max(cp) * 0.5
        B_init = np.max(cp) * 0.5
        c_init = 1.0
        d_init = 0.01
        
        param_bounds = [
            bounds['A'], bounds['B'], bounds['c'], bounds['d']
        ]
        initial_guess = [A_init, B_init, c_init, d_init]
        
        try:
            result = optimize.minimize(
                objective,
                initial_guess,
                bounds=param_bounds,
                method='L-BFGS-B'
            )
            
            if result.success:
                A_opt, B_opt, c_opt, d_opt = result.x
                model_cp = AIFModel.biexponential_aif(t, A_opt, B_opt, c_opt, d_opt)
                
                # Calculate R-squared
                ss_res = np.sum((cp - model_cp)**2)
                ss_tot = np.sum((cp - np.mean(cp))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                return {
                    'A': A_opt,
                    'B': B_opt,
                    'c': c_opt,
                    'd': d_opt,
                    'r_squared': r_squared,
                    'residual_norm': result.fun,
                    'success': True,
                    'model_fit': model_cp
                }
            else:
                logger.warning(f"AIF fitting failed: {result.message}")
                return {
                    'A': 0, 'B': 0, 'c': 0, 'd': 0,
                    'r_squared': 0,
                    'residual_norm': np.inf,
                    'success': False,
                    'model_fit': np.zeros_like(t)
                }
                
        except Exception as e:
            logger.error(f"Error in AIF fitting: {e}")
            return {
                'A': 0, 'B': 0, 'c': 0, 'd': 0,
                'r_squared': 0,
                'residual_norm': np.inf,
                'success': False,
                'model_fit': np.zeros_like(t)
            }