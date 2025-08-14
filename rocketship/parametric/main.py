"""
Main parametric fitting functions for ROCKETSHIP.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from ..utils.file_io import ImageLoader, VolumeLoader, FileParser
from ..utils.image_processing import ImageProcessor
from .models import T1Model, T2Model, T2StarModel, ADCModel


logger = logging.getLogger(__name__)


class ParametricAnalyzer:
    """Main parametric fitting analysis class."""
    
    def __init__(self, n_processes: Optional[int] = None):
        """Initialize parametric analyzer."""
        self.n_processes = n_processes or min(multiprocessing.cpu_count(), 8)
        self.models = {
            'T1_inversion_recovery': T1Model('inversion_recovery'),
            'T1_saturation_recovery': T1Model('saturation_recovery'),
            'T1_variable_flip_angle': T1Model('variable_flip_angle'),
            'T2': T2Model(),
            'T2star': T2StarModel(),
            'ADC': ADCModel()
        }
    
    def analyze_parametric_data(self, 
                              input_files: List[str],
                              parameters: List[float],
                              output_dir: str,
                              fit_type: str = 'T2',
                              rsquared_threshold: float = 0.5,
                              tr: Optional[float] = None,
                              verbose: bool = False) -> Dict[str, Any]:
        """
        Run complete parametric analysis.
        
        Args:
            input_files: List of input image files
            parameters: List of parameters (TE, TI, flip angles, b-values, etc.)
            output_dir: Output directory for results
            fit_type: Type of fit to perform
            rsquared_threshold: R² threshold for valid fits
            tr: TR value (for T1 VFA fitting)
            verbose: Verbose output
            
        Returns:
            Analysis results dictionary
        """
        logger.info(f"Starting {fit_type} parametric analysis with {len(input_files)} files")
        
        # Validate inputs
        if len(input_files) != len(parameters):
            raise ValueError(f"Number of files ({len(input_files)}) must match number of parameters ({len(parameters)})")
        
        if fit_type not in self.models:
            raise ValueError(f"Unknown fit type: {fit_type}. Available: {list(self.models.keys())}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load image series
        if verbose:
            print("Loading parametric image series...")
        
        volume, metadata = VolumeLoader.load_image_series(input_files)
        parameters = np.array(parameters)
        
        if verbose:
            print(f"Loaded volume shape: {volume.shape}")
            print(f"Parameters: {parameters}")
        
        # Sort by parameters (important for proper fitting)
        sort_indices = np.argsort(parameters)
        parameters = parameters[sort_indices]
        volume = volume[..., sort_indices]
        
        # Create brain mask for efficient processing
        brain_mask = ImageProcessor.create_brain_mask(volume[..., 0])
        
        if verbose:
            print(f"Brain mask has {np.sum(brain_mask)} voxels")
        
        # Fit parameters voxel-wise
        if verbose:
            print(f"Fitting {fit_type} model to voxels...")
        
        results = self._fit_voxel_wise(
            volume, parameters, brain_mask, fit_type, 
            rsquared_threshold, tr, verbose=verbose
        )
        
        # Save results
        self._save_results(results, output_dir, metadata, fit_type, verbose=verbose)
        
        return {
            'parameters': parameters,
            'parameter_maps': results,
            'fit_type': fit_type,
            'rsquared_threshold': rsquared_threshold
        }
    
    def _fit_voxel_wise(self, 
                       volume: np.ndarray,
                       parameters: np.ndarray,
                       brain_mask: np.ndarray,
                       fit_type: str,
                       rsquared_threshold: float,
                       tr: Optional[float] = None,
                       verbose: bool = False) -> Dict[str, np.ndarray]:
        """
        Fit parametric model to each voxel.
        """
        model = self.models[fit_type]
        nx, ny, nz, nt = volume.shape
        
        # Initialize parameter maps
        results = {}
        for param_name in model.parameter_names:
            results[param_name] = np.zeros((nx, ny, nz))
        results['r_squared'] = np.zeros((nx, ny, nz))
        
        # Get voxel coordinates for processing
        voxel_coords = []
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    if brain_mask[x, y, z]:
                        voxel_coords.append((x, y, z))
        
        total_voxels = len(voxel_coords)
        
        if verbose:
            print(f"Processing {total_voxels} voxels...")
        
        # Process voxels
        if self.n_processes > 1:
            # Parallel processing
            chunk_size = max(1, total_voxels // (self.n_processes * 4))
            
            with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
                futures = []
                
                for i in range(0, total_voxels, chunk_size):
                    chunk_coords = voxel_coords[i:i+chunk_size]
                    chunk_data = []
                    
                    for x, y, z in chunk_coords:
                        chunk_data.append(volume[x, y, z, :])
                    
                    future = executor.submit(
                        self._fit_voxel_chunk,
                        chunk_data, parameters, fit_type, rsquared_threshold, tr
                    )
                    futures.append((future, chunk_coords))
                
                # Collect results
                processed = 0
                for future, chunk_coords in futures:
                    chunk_results = future.result()
                    
                    for j, (x, y, z) in enumerate(chunk_coords):
                        if j < len(chunk_results):
                            fit_result = chunk_results[j]
                            if fit_result['success'] and fit_result['r_squared'] >= rsquared_threshold:
                                for param_name in model.parameter_names:
                                    results[param_name][x, y, z] = fit_result[param_name]
                                results['r_squared'][x, y, z] = fit_result['r_squared']
                    
                    processed += len(chunk_coords)
                    if verbose and processed % 10000 == 0:
                        progress = processed / total_voxels * 100
                        print(f"Progress: {progress:.1f}% ({processed}/{total_voxels})")
        
        else:
            # Sequential processing
            for i, (x, y, z) in enumerate(voxel_coords):
                voxel_signal = volume[x, y, z, :]
                
                # Skip if signal is too noisy or flat
                if np.std(voxel_signal) < 0.01 * np.mean(voxel_signal):
                    continue
                
                # Fit model
                fit_result = self._fit_single_voxel(
                    voxel_signal, parameters, fit_type, tr
                )
                
                # Store results if fit is good
                if fit_result['success'] and fit_result['r_squared'] >= rsquared_threshold:
                    for param_name in model.parameter_names:
                        results[param_name][x, y, z] = fit_result[param_name]
                    results['r_squared'][x, y, z] = fit_result['r_squared']
                
                if verbose and (i + 1) % 1000 == 0:
                    progress = (i + 1) / total_voxels * 100
                    print(f"Progress: {progress:.1f}% ({i + 1}/{total_voxels})")
        
        return results
    
    def _fit_voxel_chunk(self, 
                        voxel_signals: List[np.ndarray],
                        parameters: np.ndarray,
                        fit_type: str,
                        rsquared_threshold: float,
                        tr: Optional[float] = None) -> List[Dict[str, Any]]:
        """Fit a chunk of voxels (for parallel processing)."""
        results = []
        
        for voxel_signal in voxel_signals:
            # Skip if signal is too noisy or flat
            if np.std(voxel_signal) < 0.01 * np.mean(voxel_signal):
                results.append({'success': False, 'r_squared': 0})
                continue
            
            fit_result = self._fit_single_voxel(voxel_signal, parameters, fit_type, tr)
            results.append(fit_result)
        
        return results
    
    def _fit_single_voxel(self, 
                         signal: np.ndarray,
                         parameters: np.ndarray,
                         fit_type: str,
                         tr: Optional[float] = None) -> Dict[str, Any]:
        """Fit model to a single voxel."""
        model = self.models[fit_type]
        
        try:
            if fit_type.startswith('T1'):
                return model.fit(parameters, signal, tr=tr)
            elif fit_type in ['T2', 'T2star']:
                return model.fit(parameters, signal, linear_fit=True)
            elif fit_type == 'ADC':
                return model.fit(parameters, signal, linear_fit=True)
            else:
                return model.fit(parameters, signal)
                
        except Exception as e:
            logger.warning(f"Fitting failed for voxel: {e}")
            return {'success': False, 'r_squared': 0}
    
    def _save_results(self, 
                     results: Dict[str, np.ndarray],
                     output_dir: str,
                     metadata: Dict[str, Any],
                     fit_type: str,
                     verbose: bool = False) -> None:
        """Save parameter maps to NIFTI files."""
        if verbose:
            print("Saving parameter maps...")
        
        for param_name, param_map in results.items():
            # Create appropriate filename
            if param_name == 'r_squared':
                filename = f"{fit_type}_rsquared_map.nii"
            elif param_name == 'S0':
                filename = f"{fit_type}_S0_map.nii"
            else:
                filename = f"{param_name}_map.nii"
            
            output_path = os.path.join(output_dir, filename)
            
            # Apply reasonable clipping for display
            if param_name in ['T1', 'T2']:
                param_map = np.clip(param_map, 0, 5000)  # Reasonable range for T1/T2
            elif param_name == 'ADC':
                param_map = np.clip(param_map, 0, 0.005)  # Reasonable range for ADC
            elif param_name == 'r_squared':
                param_map = np.clip(param_map, 0, 1)
            
            ImageLoader.save_nifti(
                param_map, 
                output_path, 
                affine=metadata.get('affine'),
                header=metadata.get('header')
            )
            
            if verbose:
                non_zero = np.sum(param_map > 0)
                print(f"Saved {param_name} map: {output_path} ({non_zero} non-zero voxels)")


def run_parametric_fitting(config_file: Optional[str] = None,
                          input_files: Optional[List[str]] = None,
                          parameters: Optional[List[float]] = None,
                          output_dir: Optional[str] = None,
                          fit_type: str = 'T2',
                          verbose: bool = False) -> Dict[str, Any]:
    """
    Run parametric fitting from command line or script.
    
    Args:
        config_file: Configuration file path (not used yet)
        input_files: List of input image files
        parameters: List of parameters (TE, TI, etc.)
        output_dir: Output directory for results
        fit_type: Type of parametric fit
        verbose: Verbose output
        
    Returns:
        Analysis results
    """
    if input_files is None:
        # Try to find files in current directory
        input_files = FileParser.find_files_recursive(
            os.getcwd(), 
            extensions=['.nii', '.nii.gz']
        )
        
        if not input_files:
            raise ValueError("No input files specified and none found in current directory")
    
    if parameters is None:
        # Default parameters for T2 fitting (example echo times)
        if fit_type in ['T2', 'T2star']:
            parameters = [10, 20, 30, 40, 50, 60]  # Example TE values in ms
        elif fit_type.startswith('T1'):
            parameters = [50, 100, 200, 500, 1000, 2000]  # Example values
        elif fit_type == 'ADC':
            parameters = [0, 500, 1000, 1500]  # Example b-values
        else:
            raise ValueError(f"Default parameters not defined for {fit_type}")
    
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), f'{fit_type.lower()}_results')
    
    if verbose:
        print(f"Found {len(input_files)} input files")
        print(f"Parameters: {parameters}")
        print(f"Fit type: {fit_type}")
        print(f"Output directory: {output_dir}")
    
    # Initialize analyzer
    analyzer = ParametricAnalyzer()
    
    # Run analysis
    results = analyzer.analyze_parametric_data(
        input_files=input_files,
        parameters=parameters,
        output_dir=output_dir,
        fit_type=fit_type,
        verbose=verbose
    )
    
    return results