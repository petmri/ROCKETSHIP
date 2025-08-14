"""Tests for ROCKETSHIP Python package."""

import unittest
import numpy as np
from rocketship.parametric.models import T2Model, ADCModel
from rocketship.dce.models import ToftsModel, AIFModel


class TestParametricModels(unittest.TestCase):
    """Test parametric fitting models."""
    
    def test_t2_model(self):
        """Test T2 model fitting."""
        model = T2Model()
        
        # Create synthetic T2 data
        te_values = np.array([10, 20, 30, 40, 50, 60])  # Echo times in ms
        true_s0 = 1000
        true_t2 = 50
        
        # Generate noisy data
        np.random.seed(42)
        signals = true_s0 * np.exp(-te_values / true_t2)
        signals += np.random.normal(0, 10, len(signals))  # Add noise
        
        # Fit model
        result = model.fit(te_values, signals, linear_fit=True)
        
        self.assertTrue(result['success'])
        self.assertGreater(result['r_squared'], 0.8)
        self.assertAlmostEqual(result['T2'], true_t2, delta=10)
    
    def test_adc_model(self):
        """Test ADC model fitting."""
        model = ADCModel()
        
        # Create synthetic ADC data
        b_values = np.array([0, 500, 1000, 1500])  # b-values
        true_s0 = 1000
        true_adc = 0.001
        
        # Generate noisy data
        np.random.seed(42)
        signals = true_s0 * np.exp(-b_values * true_adc)
        signals += np.random.normal(0, 5, len(signals))  # Add noise
        
        # Fit model
        result = model.fit(b_values, signals, linear_fit=True)
        
        self.assertTrue(result['success'])
        self.assertGreater(result['r_squared'], 0.8)
        self.assertAlmostEqual(result['ADC'], true_adc, delta=0.0002)


class TestDCEModels(unittest.TestCase):
    """Test DCE models."""
    
    def test_aif_model(self):
        """Test AIF biexponential fitting."""
        # Create synthetic AIF data
        t = np.linspace(0, 5, 50)
        true_A = 2.0
        true_B = 1.0
        true_c = 3.0
        true_d = 0.5
        
        # Generate noisy AIF
        np.random.seed(42)
        aif = AIFModel.biexponential_aif(t, true_A, true_B, true_c, true_d)
        aif += np.random.normal(0, 0.1, len(aif))
        
        # Fit AIF
        result = AIFModel.fit_biexponential_aif(t, aif)
        
        self.assertTrue(result['success'])
        self.assertGreater(result['r_squared'], 0.8)
    
    def test_tofts_model(self):
        """Test Tofts model."""
        model = ToftsModel()
        
        # This is a simplified test - full testing would require
        # proper AIF convolution
        t = np.linspace(0, 5, 20)
        cp = np.exp(-t)  # Simple AIF
        
        # Test model function doesn't crash
        ct = model.model_function(t, 0.1, 0.3, cp)
        self.assertEqual(len(ct), len(t))


if __name__ == '__main__':
    unittest.main()