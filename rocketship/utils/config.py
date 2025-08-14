"""
Configuration and preference file parsing for ROCKETSHIP.

Python equivalent of parse_preference_file.m from the MATLAB version.
"""

import os
import configparser
import re
from typing import Dict, List, Optional, Any, Union


class PreferenceParser:
    """Parse ROCKETSHIP preference files."""
    
    def __init__(self, comment_char='%', delimiter='='):
        self.comment_char = comment_char
        self.delimiter = delimiter
    
    def parse_preference_file(self, 
                            filename: str, 
                            verbose: bool = False,
                            keywords: Optional[List[str]] = None,
                            defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse a ROCKETSHIP preferences file.
        
        Args:
            filename: Path to preferences file
            verbose: Print debug information
            keywords: List of allowed keywords to parse
            defaults: Default values for keywords
            
        Returns:
            Dictionary with parsed preferences
            
        Raises:
            FileNotFoundError: If preference file doesn't exist
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")
        
        # Initialize with defaults
        file_struct = {}
        if defaults:
            file_struct.update(defaults)
        
        if verbose:
            print(f"Parsing preference file: {filename}")
        
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith(self.comment_char):
                    continue
                
                # Parse key-value pairs
                if self.delimiter in line:
                    try:
                        key, value = line.split(self.delimiter, 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Filter by keywords if specified
                        if keywords and not self._match_keyword(key, keywords):
                            continue
                        
                        # Convert value to appropriate type
                        parsed_value = self._parse_value(value)
                        file_struct[key] = parsed_value
                        
                        if verbose:
                            print(f"  {key} = {parsed_value}")
                            
                    except Exception as e:
                        if verbose:
                            print(f"Warning: Could not parse line {line_num}: {line}")
                            print(f"  Error: {e}")
        
        return file_struct
    
    def _match_keyword(self, key: str, keywords: List[str]) -> bool:
        """Check if key matches any of the allowed keywords (case insensitive)."""
        key_lower = key.lower()
        for keyword in keywords:
            if keyword.lower() == key_lower:
                return True
        return False
    
    def _parse_value(self, value: str) -> Any:
        """Convert string value to appropriate Python type."""
        value = value.strip()
        
        # Handle empty values
        if not value:
            return ''
        
        # Try to parse as number
        try:
            # Check for scientific notation
            if 'e' in value.lower() or '^' in value:
                # Handle MATLAB scientific notation (e.g., 10^(-12))
                value = re.sub(r'10\^\(([^)]+)\)', r'1e\1', value)
                value = re.sub(r'10\^([+-]?\d+)', r'1e\1', value)
                return float(eval(value))
            
            # Try integer first
            if '.' not in value:
                return int(value)
            else:
                return float(value)
        except (ValueError, SyntaxError):
            pass
        
        # Handle boolean-like values
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # Return as string
        return value


def load_dce_preferences(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load DCE preferences from file.
    
    Args:
        config_file: Path to config file (optional)
        
    Returns:
        Dictionary with DCE preferences
    """
    if config_file is None:
        # Look for default config file
        config_file = os.path.join(os.path.dirname(__file__), '..', 'config', 'dce_preferences.txt')
        if not os.path.exists(config_file):
            # Fall back to original MATLAB file location
            config_file = 'dce_preferences.txt'
    
    parser = PreferenceParser()
    
    # Define expected DCE keywords
    dce_keywords = [
        'force_cpu', 'gpu_tolerance', 'gpu_max_n_iterations',
        'gpu_initial_value_ktrans', 'gpu_initial_value_ve', 'gpu_initial_value_vp',
        'use_matlabpool', 'autoaif_r_square_threshold', 'autoaif_end_signal_threshold',
        'autoaif_sobel_threshold', 'aif_lower_limits', 'aif_upper_limits',
        'aif_initial_values', 'aif_TolFun', 'aif_TolX', 'aif_MaxIter',
        'aif_MaxFunEvals', 'aif_Robust', 'voxel_lower_limit_ktrans',
        'voxel_upper_limit_ktrans', 'voxel_initial_value_ktrans',
        'voxel_lower_limit_ve', 'voxel_upper_limit_ve', 'voxel_initial_value_ve',
        'voxel_lower_limit_fp', 'voxel_upper_limit_fp', 'voxel_initial_value_fp',
        'voxel_lower_limit_tp', 'voxel_upper_limit_tp', 'voxel_initial_value_tp',
        'voxel_lower_limit_vp', 'voxel_upper_limit_vp', 'voxel_initial_value_vp',
        'voxel_lower_limit_tau', 'voxel_upper_limit_tau', 'voxel_initial_value_tau',
        'voxel_TolFun', 'voxel_TolX', 'voxel_MaxIter', 'voxel_MaxFunEvals',
        'voxel_Robust', 'fxr_fw'
    ]
    
    # Define defaults
    defaults = {
        'force_cpu': 0,
        'gpu_tolerance': 1e-12,
        'gpu_max_n_iterations': 200,
        'use_matlabpool': 0,
        'autoaif_r_square_threshold': 0.8,
        'aif_TolFun': 1e-20,
        'aif_TolX': 1e-23,
        'aif_MaxIter': 1000,
        'aif_MaxFunEvals': 1000,
        'aif_Robust': 'off',
        'voxel_TolFun': 1e-12,
        'voxel_TolX': 1e-6,
        'voxel_MaxIter': 50,
        'voxel_MaxFunEvals': 50,
        'voxel_Robust': 'off'
    }
    
    try:
        return parser.parse_preference_file(
            config_file, 
            keywords=dce_keywords, 
            defaults=defaults,
            verbose=False
        )
    except FileNotFoundError:
        print(f"Warning: Could not find config file {config_file}, using defaults")
        return defaults


def parse_numeric_list(value: str) -> List[float]:
    """Parse a space-separated list of numbers from a string."""
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    
    if isinstance(value, str):
        return [float(x) for x in value.split() if x.strip()]
    
    return [float(value)]


def save_preferences(preferences: Dict[str, Any], filename: str) -> None:
    """Save preferences to file in ROCKETSHIP format."""
    with open(filename, 'w') as f:
        f.write("% ROCKETSHIP Preferences File\n")
        f.write("% Generated by Python version\n\n")
        
        for key, value in preferences.items():
            if isinstance(value, (list, tuple)):
                value_str = ' '.join(str(v) for v in value)
            else:
                value_str = str(value)
            
            f.write(f"{key} = {value_str}\n")