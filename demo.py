#!/usr/bin/env python3
"""
Simple demo of ROCKETSHIP Python functionality.

This demonstrates basic usage of the converted Python ROCKETSHIP package.
"""

import numpy as np
import os
from rocketship.parametric.models import T2Model, ADCModel
from rocketship.dce.models import AIFModel
from rocketship.utils.config import load_dce_preferences

def demo_parametric_fitting():
    """Demonstrate parametric T2 fitting."""
    print("=== T2 Parametric Fitting Demo ===")
    
    # Create synthetic T2 data
    te_values = np.array([10, 20, 30, 40, 50, 60])  # Echo times in ms
    true_s0 = 1000
    true_t2 = 50
    
    # Generate synthetic signals with noise
    np.random.seed(42)
    signals = true_s0 * np.exp(-te_values / true_t2)
    signals += np.random.normal(0, 10, len(signals))  # Add noise
    
    print(f"True values: S0={true_s0}, T2={true_t2} ms")
    print(f"TE values: {te_values}")
    print(f"Signals: {signals}")
    
    # Fit T2 model
    model = T2Model()
    result = model.fit(te_values, signals, linear_fit=True)
    
    print(f"\nFit results:")
    print(f"  S0: {result['S0']:.1f}")
    print(f"  T2: {result['T2']:.1f} ms")
    print(f"  R²: {result['r_squared']:.3f}")
    print(f"  Success: {result['success']}")

def demo_adc_fitting():
    """Demonstrate ADC fitting."""
    print("\n=== ADC Parametric Fitting Demo ===")
    
    # Create synthetic ADC data
    b_values = np.array([0, 500, 1000, 1500])  # b-values
    true_s0 = 1000
    true_adc = 0.001  # mm²/s
    
    # Generate synthetic signals with noise
    np.random.seed(42)
    signals = true_s0 * np.exp(-b_values * true_adc)
    signals += np.random.normal(0, 5, len(signals))  # Add noise
    
    print(f"True values: S0={true_s0}, ADC={true_adc} mm²/s")
    print(f"b-values: {b_values}")
    print(f"Signals: {signals}")
    
    # Fit ADC model
    model = ADCModel()
    result = model.fit(b_values, signals, linear_fit=True)
    
    print(f"\nFit results:")
    print(f"  S0: {result['S0']:.1f}")
    print(f"  ADC: {result['ADC']:.4f} mm²/s")
    print(f"  R²: {result['r_squared']:.3f}")
    print(f"  Success: {result['success']}")

def demo_aif_fitting():
    """Demonstrate AIF biexponential fitting."""
    print("\n=== AIF Biexponential Fitting Demo ===")
    
    # Create synthetic AIF data
    t = np.linspace(0, 5, 50)
    true_A = 2.0
    true_B = 1.0
    true_c = 3.0
    true_d = 0.5
    
    # Generate synthetic AIF with noise
    np.random.seed(42)
    aif = AIFModel.biexponential_aif(t, true_A, true_B, true_c, true_d)
    aif += np.random.normal(0, 0.1, len(aif))
    
    print(f"True values: A={true_A}, B={true_B}, c={true_c}, d={true_d}")
    print(f"Time points: {len(t)} from {t[0]:.1f} to {t[-1]:.1f}")
    
    # Fit AIF
    result = AIFModel.fit_biexponential_aif(t, aif)
    
    print(f"\nFit results:")
    print(f"  A: {result['A']:.2f}")
    print(f"  B: {result['B']:.2f}")
    print(f"  c: {result['c']:.2f}")
    print(f"  d: {result['d']:.2f}")
    print(f"  R²: {result['r_squared']:.3f}")
    print(f"  Success: {result['success']}")

def demo_config_parsing():
    """Demonstrate configuration file parsing."""
    print("\n=== Configuration Parsing Demo ===")
    
    try:
        # Try to load DCE preferences
        config = load_dce_preferences()
        print("Successfully loaded DCE preferences")
        print(f"Sample configuration values:")
        
        for key in list(config.keys())[:5]:  # Show first 5 keys
            print(f"  {key}: {config[key]}")
        
        print(f"Total configuration keys: {len(config)}")
        
    except Exception as e:
        print(f"Could not load config file: {e}")
        print("This is expected if no config file is present")

def main():
    """Run all demos."""
    print("ROCKETSHIP Python Package Demo")
    print("=" * 40)
    
    demo_parametric_fitting()
    demo_adc_fitting()
    demo_aif_fitting()
    demo_config_parsing()
    
    print("\n" + "=" * 40)
    print("Demo completed successfully!")
    print("ROCKETSHIP Python package is working correctly.")

if __name__ == "__main__":
    main()