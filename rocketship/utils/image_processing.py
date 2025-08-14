"""
Image processing utilities for ROCKETSHIP.

Basic image processing functions for medical imaging data.
"""

import numpy as np
import scipy.ndimage as ndi
from scipy import signal
from typing import Tuple, Optional, Union
import logging


logger = logging.getLogger(__name__)


class ImageProcessor:
    """Image processing utilities."""
    
    @staticmethod
    def smooth_image(image: np.ndarray, 
                    sigma: Union[float, Tuple[float, ...]] = 1.0,
                    mode: str = 'reflect') -> np.ndarray:
        """
        Apply Gaussian smoothing to image.
        
        Args:
            image: Input image
            sigma: Standard deviation for Gaussian smoothing
            mode: Boundary condition mode
            
        Returns:
            Smoothed image
        """
        return ndi.gaussian_filter(image, sigma=sigma, mode=mode)
    
    @staticmethod
    def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply binary mask to image.
        
        Args:
            image: Input image
            mask: Binary mask
            
        Returns:
            Masked image
        """
        return image * mask
    
    @staticmethod
    def create_brain_mask(image: np.ndarray, 
                         threshold_factor: float = 0.1) -> np.ndarray:
        """
        Create simple brain mask using intensity thresholding.
        
        Args:
            image: Input brain image
            threshold_factor: Fraction of max intensity for threshold
            
        Returns:
            Binary brain mask
        """
        # Simple intensity-based masking
        threshold = np.max(image) * threshold_factor
        mask = image > threshold
        
        # Clean up mask with morphological operations
        mask = ndi.binary_fill_holes(mask)
        mask = ndi.binary_opening(mask, structure=np.ones((3, 3, 3)))
        mask = ndi.binary_closing(mask, structure=np.ones((5, 5, 5)))
        
        return mask.astype(np.uint8)
    
    @staticmethod
    def normalize_image(image: np.ndarray, 
                       method: str = 'minmax',
                       mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Normalize image intensities.
        
        Args:
            image: Input image
            method: Normalization method ('minmax', 'zscore', 'percent')
            mask: Optional mask for calculating statistics
            
        Returns:
            Normalized image
        """
        if mask is not None:
            masked_data = image[mask > 0]
        else:
            masked_data = image
        
        if method == 'minmax':
            min_val = np.min(masked_data)
            max_val = np.max(masked_data)
            if max_val > min_val:
                return (image - min_val) / (max_val - min_val)
            else:
                return image
        
        elif method == 'zscore':
            mean_val = np.mean(masked_data)
            std_val = np.std(masked_data)
            if std_val > 0:
                return (image - mean_val) / std_val
            else:
                return image - mean_val
        
        elif method == 'percent':
            # Normalize to 95th percentile
            p95 = np.percentile(masked_data, 95)
            if p95 > 0:
                return image / p95
            else:
                return image
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def calculate_snr(signal_image: np.ndarray, 
                     noise_image: Optional[np.ndarray] = None,
                     signal_mask: Optional[np.ndarray] = None,
                     noise_mask: Optional[np.ndarray] = None) -> float:
        """
        Calculate signal-to-noise ratio.
        
        Args:
            signal_image: Signal image
            noise_image: Noise image (optional, uses signal image if None)
            signal_mask: Mask for signal region
            noise_mask: Mask for noise region
            
        Returns:
            SNR value
        """
        if noise_image is None:
            noise_image = signal_image
        
        if signal_mask is not None:
            signal_data = signal_image[signal_mask > 0]
        else:
            signal_data = signal_image.flatten()
        
        if noise_mask is not None:
            noise_data = noise_image[noise_mask > 0]
        else:
            # Use corners of image as noise estimate
            h, w = signal_image.shape[:2]
            corner_size = min(h, w) // 10
            noise_data = np.concatenate([
                signal_image[:corner_size, :corner_size].flatten(),
                signal_image[-corner_size:, :corner_size].flatten(),
                signal_image[:corner_size, -corner_size:].flatten(),
                signal_image[-corner_size:, -corner_size:].flatten()
            ])
        
        signal_mean = np.mean(signal_data)
        noise_std = np.std(noise_data)
        
        if noise_std > 0:
            return signal_mean / noise_std
        else:
            return float('inf')
    
    @staticmethod
    def estimate_noise_level(image: np.ndarray, 
                           method: str = 'background') -> float:
        """
        Estimate noise level in image.
        
        Args:
            image: Input image
            method: Method for noise estimation
            
        Returns:
            Estimated noise standard deviation
        """
        if method == 'background':
            # Use background corners
            h, w = image.shape[:2]
            corner_size = min(h, w) // 10
            corners = np.concatenate([
                image[:corner_size, :corner_size].flatten(),
                image[-corner_size:, :corner_size].flatten(),
                image[:corner_size, -corner_size:].flatten(),
                image[-corner_size:, -corner_size:].flatten()
            ])
            return np.std(corners)
        
        elif method == 'laplacian':
            # Use Laplacian method
            if image.ndim == 2:
                laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
            else:
                # 3D Laplacian
                laplacian = np.zeros((3, 3, 3))
                laplacian[1, 1, 1] = 6
                laplacian[1, 1, 0] = laplacian[1, 1, 2] = -1
                laplacian[1, 0, 1] = laplacian[1, 2, 1] = -1
                laplacian[0, 1, 1] = laplacian[2, 1, 1] = -1
            
            filtered = ndi.convolve(image, laplacian)
            return np.std(filtered) / np.sqrt(2)
        
        else:
            raise ValueError(f"Unknown noise estimation method: {method}")


class SignalAnalyzer:
    """Analyze signal time courses."""
    
    @staticmethod
    def calculate_auc(time_points: np.ndarray, 
                     signal: np.ndarray,
                     baseline_end: Optional[int] = None) -> float:
        """
        Calculate area under curve.
        
        Args:
            time_points: Time points array
            signal: Signal values
            baseline_end: Index of baseline end (optional)
            
        Returns:
            AUC value
        """
        if baseline_end is not None:
            baseline_mean = np.mean(signal[:baseline_end])
            signal_corrected = signal - baseline_mean
        else:
            signal_corrected = signal
        
        return np.trapz(signal_corrected, time_points)
    
    @staticmethod
    def find_peak_enhancement(time_points: np.ndarray, 
                            signal: np.ndarray,
                            baseline_end: Optional[int] = None) -> Tuple[float, int, float]:
        """
        Find peak enhancement in signal.
        
        Args:
            time_points: Time points array
            signal: Signal values
            baseline_end: Index of baseline end
            
        Returns:
            Tuple of (peak_value, peak_index, peak_time)
        """
        if baseline_end is not None:
            baseline_mean = np.mean(signal[:baseline_end])
            search_signal = signal[baseline_end:]
            search_times = time_points[baseline_end:]
            
            peak_idx_relative = np.argmax(search_signal)
            peak_idx = peak_idx_relative + baseline_end
            peak_enhancement = search_signal[peak_idx_relative] - baseline_mean
        else:
            peak_idx = np.argmax(signal)
            peak_enhancement = signal[peak_idx]
        
        return peak_enhancement, peak_idx, time_points[peak_idx]
    
    @staticmethod
    def calculate_percent_enhancement(pre_signal: np.ndarray, 
                                   post_signal: np.ndarray) -> np.ndarray:
        """
        Calculate percent enhancement.
        
        Args:
            pre_signal: Pre-contrast signal
            post_signal: Post-contrast signal
            
        Returns:
            Percent enhancement array
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            enhancement = (post_signal - pre_signal) / pre_signal * 100
            enhancement[~np.isfinite(enhancement)] = 0
        
        return enhancement
    
    @staticmethod
    def smooth_time_series(signal: np.ndarray, 
                          window_size: int = 3,
                          method: str = 'gaussian') -> np.ndarray:
        """
        Smooth time series data.
        
        Args:
            signal: Input signal
            window_size: Size of smoothing window
            method: Smoothing method
            
        Returns:
            Smoothed signal
        """
        if method == 'gaussian':
            sigma = window_size / 3.0
            return ndi.gaussian_filter1d(signal, sigma=sigma)
        
        elif method == 'median':
            return ndi.median_filter(signal, size=window_size)
        
        elif method == 'moving_average':
            kernel = np.ones(window_size) / window_size
            return np.convolve(signal, kernel, mode='same')
        
        else:
            raise ValueError(f"Unknown smoothing method: {method}")


class ROIAnalyzer:
    """Analyze regions of interest."""
    
    @staticmethod
    def extract_roi_signal(image: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        """
        Extract signal from ROI.
        
        Args:
            image: Input image (can be 4D time series)
            roi_mask: Binary ROI mask
            
        Returns:
            Mean signal from ROI
        """
        if image.ndim == 3:
            # 3D image
            return np.mean(image[roi_mask > 0])
        elif image.ndim == 4:
            # 4D time series
            roi_signals = []
            for t in range(image.shape[-1]):
                roi_signals.append(np.mean(image[..., t][roi_mask > 0]))
            return np.array(roi_signals)
        else:
            raise ValueError("Image must be 3D or 4D")
    
    @staticmethod
    def create_spherical_roi(center: Tuple[int, int, int], 
                           radius: float,
                           image_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Create spherical ROI mask.
        
        Args:
            center: ROI center coordinates
            radius: ROI radius in voxels
            image_shape: Shape of image
            
        Returns:
            Binary ROI mask
        """
        x, y, z = np.meshgrid(
            np.arange(image_shape[0]),
            np.arange(image_shape[1]),
            np.arange(image_shape[2]),
            indexing='ij'
        )
        
        distance = np.sqrt(
            (x - center[0])**2 + 
            (y - center[1])**2 + 
            (z - center[2])**2
        )
        
        return (distance <= radius).astype(np.uint8)