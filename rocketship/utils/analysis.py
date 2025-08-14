"""
Analysis and visualization utilities for ROCKETSHIP.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import logging

from .file_io import ImageLoader, FileParser
from .image_processing import ImageProcessor, ROIAnalyzer


logger = logging.getLogger(__name__)


class ResultAnalyzer:
    """Analyze and visualize ROCKETSHIP results."""
    
    def __init__(self):
        self.color_maps = {
            'T1': 'viridis',
            'T2': 'plasma',
            'T2star': 'plasma',
            'ADC': 'magma',
            'Ktrans': 'hot',
            've': 'cool',
            'vp': 'autumn',
            'r_squared': 'RdYlBu_r'
        }
    
    def load_parameter_maps(self, results_dir: str) -> Dict[str, np.ndarray]:
        """
        Load parameter maps from results directory.
        
        Args:
            results_dir: Directory containing NIFTI parameter maps
            
        Returns:
            Dictionary of parameter maps
        """
        parameter_maps = {}
        
        # Find all NIFTI files in results directory
        nifti_files = FileParser.find_files_recursive(
            results_dir, 
            extensions=['.nii', '.nii.gz']
        )
        
        for file_path in nifti_files:
            filename = os.path.basename(file_path)
            param_name = self._extract_parameter_name(filename)
            
            if param_name:
                try:
                    data, header, affine = ImageLoader.load_nifti(file_path)
                    parameter_maps[param_name] = {
                        'data': data,
                        'header': header,
                        'affine': affine,
                        'filename': file_path
                    }
                    logger.info(f"Loaded {param_name} map from {filename}")
                except Exception as e:
                    logger.warning(f"Could not load {filename}: {e}")
        
        return parameter_maps
    
    def _extract_parameter_name(self, filename: str) -> Optional[str]:
        """Extract parameter name from filename."""
        filename = filename.lower()
        
        if 'ktrans' in filename:
            return 'Ktrans'
        elif 've_map' in filename:
            return 've'
        elif 'vp_map' in filename:
            return 'vp'
        elif 't1_map' in filename or filename.startswith('t1_'):
            return 'T1'
        elif 't2star' in filename or 't2_star' in filename:
            return 'T2star'
        elif 't2_map' in filename or filename.startswith('t2_'):
            return 'T2'
        elif 'adc' in filename:
            return 'ADC'
        elif 'rsquared' in filename or 'r_squared' in filename:
            return 'r_squared'
        elif 's0' in filename:
            return 'S0'
        
        return None
    
    def create_summary_plots(self, 
                           parameter_maps: Dict[str, Any],
                           output_dir: str,
                           slice_idx: Optional[int] = None) -> None:
        """
        Create summary plots of parameter maps.
        
        Args:
            parameter_maps: Dictionary of parameter maps
            output_dir: Output directory for plots
            slice_idx: Slice index to display (middle slice if None)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for param_name, map_info in parameter_maps.items():
            data = map_info['data']
            
            if data.ndim == 3:
                # Select middle slice if not specified
                if slice_idx is None:
                    slice_idx = data.shape[2] // 2
                
                slice_data = data[:, :, slice_idx]
                
                # Create plot
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                
                # Determine color scale
                if param_name in ['T1', 'T2', 'T2star']:
                    vmax = np.percentile(slice_data[slice_data > 0], 95)
                    vmin = 0
                elif param_name == 'ADC':
                    vmax = 0.003
                    vmin = 0
                elif param_name in ['Ktrans', 've', 'vp']:
                    vmax = np.percentile(slice_data[slice_data > 0], 95)
                    vmin = 0
                elif param_name == 'r_squared':
                    vmax = 1
                    vmin = 0
                else:
                    vmax = np.percentile(slice_data[slice_data > 0], 95)
                    vmin = 0
                
                # Plot
                cmap = self.color_maps.get(param_name, 'viridis')
                im = ax.imshow(slice_data.T, cmap=cmap, vmin=vmin, vmax=vmax, 
                              origin='lower', aspect='equal')
                
                ax.set_title(f'{param_name} Map (Slice {slice_idx})')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                
                # Colorbar
                cbar = plt.colorbar(im, ax=ax)
                
                # Set units
                if param_name in ['T1', 'T2', 'T2star']:
                    cbar.set_label(f'{param_name} (ms)')
                elif param_name == 'ADC':
                    cbar.set_label('ADC (mm²/s)')
                elif param_name == 'Ktrans':
                    cbar.set_label('Ktrans (min⁻¹)')
                elif param_name == 'r_squared':
                    cbar.set_label('R²')
                else:
                    cbar.set_label(param_name)
                
                # Save plot
                output_filename = os.path.join(output_dir, f'{param_name}_map.png')
                plt.savefig(output_filename, dpi=150, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Saved {param_name} plot: {output_filename}")
    
    def calculate_statistics(self, 
                           parameter_maps: Dict[str, Any],
                           roi_mask: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for parameter maps.
        
        Args:
            parameter_maps: Dictionary of parameter maps
            roi_mask: Optional ROI mask for regional analysis
            
        Returns:
            Dictionary of statistics for each parameter
        """
        stats = {}
        
        for param_name, map_info in parameter_maps.items():
            data = map_info['data']
            
            if roi_mask is not None:
                # Regional statistics
                roi_data = data[roi_mask > 0]
            else:
                # Whole-brain statistics (exclude zeros)
                roi_data = data[data > 0]
            
            if len(roi_data) > 0:
                stats[param_name] = {
                    'mean': float(np.mean(roi_data)),
                    'std': float(np.std(roi_data)),
                    'median': float(np.median(roi_data)),
                    'min': float(np.min(roi_data)),
                    'max': float(np.max(roi_data)),
                    'p25': float(np.percentile(roi_data, 25)),
                    'p75': float(np.percentile(roi_data, 75)),
                    'n_voxels': len(roi_data)
                }
            else:
                stats[param_name] = {
                    'mean': 0, 'std': 0, 'median': 0, 'min': 0, 'max': 0,
                    'p25': 0, 'p75': 0, 'n_voxels': 0
                }
        
        return stats
    
    def create_histogram_plots(self, 
                             parameter_maps: Dict[str, Any],
                             output_dir: str,
                             roi_mask: Optional[np.ndarray] = None) -> None:
        """
        Create histogram plots for parameter maps.
        
        Args:
            parameter_maps: Dictionary of parameter maps
            output_dir: Output directory for plots
            roi_mask: Optional ROI mask
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for param_name, map_info in parameter_maps.items():
            data = map_info['data']
            
            if roi_mask is not None:
                plot_data = data[roi_mask > 0]
                title_suffix = " (ROI)"
            else:
                plot_data = data[data > 0]
                title_suffix = " (Non-zero voxels)"
            
            if len(plot_data) == 0:
                continue
            
            # Create histogram
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            # Determine bins
            if param_name in ['T1', 'T2', 'T2star']:
                bins = np.linspace(0, np.percentile(plot_data, 95), 50)
            elif param_name == 'ADC':
                bins = np.linspace(0, 0.003, 50)
            elif param_name == 'r_squared':
                bins = np.linspace(0, 1, 50)
            else:
                bins = 50
            
            ax.hist(plot_data, bins=bins, alpha=0.7, edgecolor='black')
            ax.set_xlabel(param_name)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{param_name} Distribution{title_suffix}')
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            mean_val = np.mean(plot_data)
            std_val = np.std(plot_data)
            ax.axvline(mean_val, color='red', linestyle='--', 
                      label=f'Mean: {mean_val:.3f}')
            ax.legend()
            
            # Save plot
            output_filename = os.path.join(output_dir, f'{param_name}_histogram.png')
            plt.savefig(output_filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved {param_name} histogram: {output_filename}")


def run_analysis(input_dir: str,
                output_dir: Optional[str] = None,
                verbose: bool = False) -> Dict[str, Any]:
    """
    Run analysis of ROCKETSHIP results.
    
    Args:
        input_dir: Directory containing parameter maps
        output_dir: Output directory for analysis results
        verbose: Verbose output
        
    Returns:
        Analysis results
    """
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'analysis')
    
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"Analyzing results from: {input_dir}")
        print(f"Output directory: {output_dir}")
    
    # Initialize analyzer
    analyzer = ResultAnalyzer()
    
    # Load parameter maps
    if verbose:
        print("Loading parameter maps...")
    
    parameter_maps = analyzer.load_parameter_maps(input_dir)
    
    if not parameter_maps:
        raise ValueError(f"No parameter maps found in {input_dir}")
    
    if verbose:
        print(f"Loaded {len(parameter_maps)} parameter maps:")
        for param_name in parameter_maps:
            shape = parameter_maps[param_name]['data'].shape
            print(f"  {param_name}: {shape}")
    
    # Calculate statistics
    if verbose:
        print("Calculating statistics...")
    
    stats = analyzer.calculate_statistics(parameter_maps)
    
    # Create plots
    if verbose:
        print("Creating summary plots...")
    
    plot_dir = os.path.join(output_dir, 'plots')
    analyzer.create_summary_plots(parameter_maps, plot_dir)
    analyzer.create_histogram_plots(parameter_maps, plot_dir)
    
    # Save statistics to file
    stats_file = os.path.join(output_dir, 'statistics.txt')
    with open(stats_file, 'w') as f:
        f.write("ROCKETSHIP Analysis Results\n")
        f.write("=" * 40 + "\n\n")
        
        for param_name, param_stats in stats.items():
            f.write(f"{param_name}:\n")
            f.write(f"  Mean ± SD: {param_stats['mean']:.3f} ± {param_stats['std']:.3f}\n")
            f.write(f"  Median: {param_stats['median']:.3f}\n")
            f.write(f"  Range: {param_stats['min']:.3f} - {param_stats['max']:.3f}\n")
            f.write(f"  IQR: {param_stats['p25']:.3f} - {param_stats['p75']:.3f}\n")
            f.write(f"  N voxels: {param_stats['n_voxels']}\n\n")
    
    if verbose:
        print(f"Analysis complete. Results saved to: {output_dir}")
        print(f"Statistics saved to: {stats_file}")
    
    return {
        'parameter_maps': parameter_maps,
        'statistics': stats,
        'output_dir': output_dir
    }