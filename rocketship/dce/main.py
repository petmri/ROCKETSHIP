"""
Main DCE-MRI analysis functions for ROCKETSHIP.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

from ..utils.config import load_dce_preferences
from ..utils.file_io import ImageLoader, VolumeLoader, FileParser
from ..utils.image_processing import ImageProcessor, ROIAnalyzer
from .models import ToftsModel, ExtendedToftsModel, PatlakModel, AIFModel


logger = logging.getLogger(__name__)


class DCEAnalyzer:
    """Main DCE-MRI analysis class."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize DCE analyzer with configuration."""
        self.config = load_dce_preferences(config_file)
        self.models = {
            'tofts': ToftsModel(),
            'extended_tofts': ExtendedToftsModel(),
            'patlak': PatlakModel()
        }
    
    def analyze_dce_data(self, 
                        input_files: List[str],
                        output_dir: str,
                        aif_roi: Optional[np.ndarray] = None,
                        model_type: str = 'tofts',
                        verbose: bool = False) -> Dict[str, Any]:
        """
        Run complete DCE-MRI analysis.
        
        Args:
            input_files: List of DCE time series files
            output_dir: Output directory for results
            aif_roi: ROI mask for AIF extraction (optional)
            model_type: Pharmacokinetic model to use
            verbose: Verbose output
            
        Returns:
            Analysis results dictionary
        """
        logger.info(f"Starting DCE analysis with {len(input_files)} files")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load DCE time series
        if verbose:
            print("Loading DCE time series...")
        
        volume, metadata = VolumeLoader.load_image_series(input_files)
        
        if verbose:
            print(f"Loaded volume shape: {volume.shape}")
        
        # Extract AIF
        if aif_roi is not None:
            aif_signal = ROIAnalyzer.extract_roi_signal(volume, aif_roi)
        else:
            # Auto-detect AIF (simplified version)
            aif_signal = self._auto_detect_aif(volume, verbose=verbose)
        
        if verbose:
            print(f"AIF signal shape: {aif_signal.shape}")
        
        # Generate time points (assuming consistent timing)
        n_timepoints = volume.shape[-1]
        time_points = np.arange(n_timepoints, dtype=np.float64)
        
        # Fit AIF model
        aif_fit = AIFModel.fit_biexponential_aif(time_points, aif_signal)
        
        if verbose and aif_fit['success']:
            print(f"AIF fit R² = {aif_fit['r_squared']:.3f}")
        
        # Fit pharmacokinetic model to each voxel
        if verbose:
            print(f"Fitting {model_type} model to voxels...")
        
        results = self._fit_voxel_wise(
            volume, time_points, aif_signal, 
            model_type, verbose=verbose
        )
        
        # Save results
        self._save_results(results, output_dir, metadata, verbose=verbose)
        
        return {
            'aif_signal': aif_signal,
            'aif_fit': aif_fit,
            'time_points': time_points,
            'parameter_maps': results,
            'config': self.config
        }
    
    def _auto_detect_aif(self, volume: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Auto-detect AIF from DCE volume.
        
        This is a simplified version of the auto AIF detection.
        """
        if verbose:
            print("Auto-detecting AIF...")
        
        # Calculate enhancement for each voxel
        baseline = np.mean(volume[..., :3], axis=-1)  # First 3 timepoints as baseline
        peak_enhancement = np.max(volume, axis=-1) - baseline
        
        # Find voxels with high enhancement and early arrival
        enhancement_threshold = np.percentile(peak_enhancement, 95)
        high_enhancement_mask = peak_enhancement > enhancement_threshold
        
        # Find time-to-peak for high enhancement voxels
        candidate_signals = []
        for z in range(volume.shape[2]):
            for y in range(volume.shape[1]):
                for x in range(volume.shape[0]):
                    if high_enhancement_mask[x, y, z]:
                        signal = volume[x, y, z, :]
                        candidate_signals.append(signal)
        
        if not candidate_signals:
            # Fallback: use center of volume
            center = tuple(s // 2 for s in volume.shape[:3])
            return volume[center[0], center[1], center[2], :]
        
        # Select signal with earliest and highest peak
        candidate_signals = np.array(candidate_signals)
        peak_times = np.argmax(candidate_signals, axis=1)
        early_peak_idx = np.argmin(peak_times)
        
        return candidate_signals[early_peak_idx]
    
    def _fit_voxel_wise(self, 
                       volume: np.ndarray,
                       time_points: np.ndarray,
                       aif_signal: np.ndarray,
                       model_type: str,
                       verbose: bool = False) -> Dict[str, np.ndarray]:
        """
        Fit pharmacokinetic model to each voxel.
        """
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = self.models[model_type]
        
        # Initialize parameter maps
        nx, ny, nz, nt = volume.shape
        results = {}
        
        for param_name in model.parameter_names:
            results[param_name] = np.zeros((nx, ny, nz))
        
        results['r_squared'] = np.zeros((nx, ny, nz))
        
        # Create simple brain mask (optional optimization)
        brain_mask = ImageProcessor.create_brain_mask(volume[..., 0])
        
        total_voxels = np.sum(brain_mask)
        processed_voxels = 0
        
        # Fit each voxel
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    if not brain_mask[x, y, z]:
                        continue
                    
                    voxel_signal = volume[x, y, z, :]
                    
                    # Skip if signal is too noisy or flat
                    if np.std(voxel_signal) < 0.01 * np.mean(voxel_signal):
                        continue
                    
                    # Fit model
                    fit_result = model.fit(time_points, voxel_signal, aif_signal)
                    
                    # Store results
                    if fit_result['success']:
                        for param_name in model.parameter_names:
                            results[param_name][x, y, z] = fit_result[param_name]
                        results['r_squared'][x, y, z] = fit_result['r_squared']
                    
                    processed_voxels += 1
                    if verbose and processed_voxels % 1000 == 0:
                        progress = processed_voxels / total_voxels * 100
                        print(f"Progress: {progress:.1f}% ({processed_voxels}/{total_voxels})")
        
        return results
    
    def _save_results(self, 
                     results: Dict[str, np.ndarray],
                     output_dir: str,
                     metadata: Dict[str, Any],
                     verbose: bool = False) -> None:
        """Save parameter maps to NIFTI files."""
        if verbose:
            print("Saving parameter maps...")
        
        for param_name, param_map in results.items():
            output_filename = os.path.join(output_dir, f"{param_name}_map.nii")
            ImageLoader.save_nifti(
                param_map, 
                output_filename, 
                affine=metadata.get('affine'),
                header=metadata.get('header')
            )
            
            if verbose:
                print(f"Saved {param_name} map: {output_filename}")


def run_dce_analysis(config_file: Optional[str] = None,
                    input_dir: Optional[str] = None,
                    output_dir: Optional[str] = None,
                    verbose: bool = False) -> Dict[str, Any]:
    """
    Run DCE analysis from command line or script.
    
    Args:
        config_file: Configuration file path
        input_dir: Input directory containing DCE files
        output_dir: Output directory for results
        verbose: Verbose output
        
    Returns:
        Analysis results
    """
    if input_dir is None:
        input_dir = os.getcwd()
    
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'dce_results')
    
    # Find DCE files
    input_files = FileParser.find_files_recursive(
        input_dir, 
        extensions=['.nii', '.nii.gz']
    )
    
    if not input_files:
        raise ValueError(f"No DCE files found in {input_dir}")
    
    if verbose:
        print(f"Found {len(input_files)} DCE files")
        for f in input_files[:5]:  # Show first 5
            print(f"  {f}")
        if len(input_files) > 5:
            print(f"  ... and {len(input_files) - 5} more")
    
    # Initialize analyzer
    analyzer = DCEAnalyzer(config_file)
    
    # Run analysis
    results = analyzer.analyze_dce_data(
        input_files=input_files,
        output_dir=output_dir,
        model_type='tofts',  # Default to Tofts model
        verbose=verbose
    )
    
    return results