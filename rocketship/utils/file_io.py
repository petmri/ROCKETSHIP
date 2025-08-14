"""
File I/O utilities for ROCKETSHIP.

Handles NIFTI and DICOM file operations, replacing MATLAB's niftitools functionality.
"""

import os
import numpy as np
import nibabel as nib
import pydicom
from typing import Tuple, List, Optional, Union, Dict, Any
import logging


logger = logging.getLogger(__name__)


class ImageLoader:
    """Load medical imaging files (NIFTI, DICOM)."""
    
    @staticmethod
    def load_nifti(filename: str) -> Tuple[np.ndarray, nib.Nifti1Header, np.ndarray]:
        """
        Load NIFTI file and return image data, header, and affine matrix.
        
        Args:
            filename: Path to NIFTI file (.nii or .nii.gz)
            
        Returns:
            Tuple of (image_data, header, affine)
        """
        try:
            nii = nib.load(filename)
            return nii.get_fdata(), nii.header, nii.affine
        except Exception as e:
            raise IOError(f"Could not load NIFTI file {filename}: {e}")
    
    @staticmethod
    def save_nifti(data: np.ndarray, 
                   filename: str, 
                   affine: Optional[np.ndarray] = None,
                   header: Optional[nib.Nifti1Header] = None) -> None:
        """
        Save array as NIFTI file.
        
        Args:
            data: Image data array
            filename: Output filename
            affine: Affine transformation matrix
            header: NIFTI header (optional)
        """
        if affine is None:
            affine = np.eye(4)
        
        nii = nib.Nifti1Image(data, affine, header)
        nib.save(nii, filename)
        logger.info(f"Saved NIFTI file: {filename}")
    
    @staticmethod
    def is_nifti(filename: str) -> bool:
        """Check if file is a NIFTI file."""
        return filename.lower().endswith(('.nii', '.nii.gz'))
    
    @staticmethod
    def is_dicom(filename: str) -> bool:
        """Check if file is a DICOM file."""
        if filename.lower().endswith('.dcm'):
            return True
        
        try:
            pydicom.dcmread(filename, stop_before_pixels=True)
            return True
        except:
            return False
    
    @staticmethod
    def load_dicom(filename: str) -> Tuple[np.ndarray, pydicom.Dataset]:
        """
        Load DICOM file.
        
        Args:
            filename: Path to DICOM file
            
        Returns:
            Tuple of (image_data, dicom_dataset)
        """
        try:
            ds = pydicom.dcmread(filename)
            return ds.pixel_array.astype(np.float64), ds
        except Exception as e:
            raise IOError(f"Could not load DICOM file {filename}: {e}")


class FileParser:
    """Parse and organize image files."""
    
    @staticmethod
    def natural_sort(file_list: List[str]) -> List[str]:
        """
        Sort files in natural order (e.g., file1, file2, file10).
        
        Equivalent to natORDER.m from MATLAB version.
        """
        import re
        
        def natural_key(text):
            return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]
        
        return sorted(file_list, key=natural_key)
    
    @staticmethod
    def find_files_recursive(directory: str, 
                           pattern: str = "*.nii*",
                           extensions: Optional[List[str]] = None) -> List[str]:
        """
        Find files recursively in directory.
        
        Args:
            directory: Root directory to search
            pattern: File pattern (not used if extensions specified)
            extensions: List of file extensions to match
            
        Returns:
            List of found file paths
        """
        import glob
        
        if extensions is None:
            extensions = ['.nii', '.nii.gz', '.dcm']
        
        files = []
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                if any(filename.lower().endswith(ext) for ext in extensions):
                    files.append(os.path.join(root, filename))
        
        return FileParser.natural_sort(files)
    
    @staticmethod
    def group_files_by_series(file_list: List[str]) -> Dict[str, List[str]]:
        """
        Group files by series (useful for DICOM series).
        
        Args:
            file_list: List of file paths
            
        Returns:
            Dictionary mapping series ID to file list
        """
        series_dict = {}
        
        for filename in file_list:
            try:
                if ImageLoader.is_dicom(filename):
                    ds = pydicom.dcmread(filename, stop_before_pixels=True)
                    series_id = getattr(ds, 'SeriesInstanceUID', 'unknown')
                else:
                    # For NIFTI, use directory as series identifier
                    series_id = os.path.dirname(filename)
                
                if series_id not in series_dict:
                    series_dict[series_id] = []
                series_dict[series_id].append(filename)
                
            except Exception as e:
                logger.warning(f"Could not process file {filename}: {e}")
                series_id = 'unknown'
                if series_id not in series_dict:
                    series_dict[series_id] = []
                series_dict[series_id].append(filename)
        
        # Sort files within each series
        for series_id in series_dict:
            series_dict[series_id] = FileParser.natural_sort(series_dict[series_id])
        
        return series_dict
    
    @staticmethod
    def get_file_prefix(filename: str) -> str:
        """Get file prefix (filename without extension)."""
        basename = os.path.basename(filename)
        if basename.endswith('.nii.gz'):
            return basename[:-7]
        elif basename.endswith('.nii'):
            return basename[:-4]
        elif basename.endswith('.dcm'):
            return basename[:-4]
        else:
            return os.path.splitext(basename)[0]
    
    @staticmethod
    def generate_output_filename(input_filename: str, 
                                suffix: str,
                                output_dir: Optional[str] = None,
                                extension: str = '.nii') -> str:
        """
        Generate output filename based on input filename.
        
        Args:
            input_filename: Input file path
            suffix: Suffix to add to filename
            output_dir: Output directory (optional)
            extension: File extension for output
            
        Returns:
            Output file path
        """
        prefix = FileParser.get_file_prefix(input_filename)
        output_name = f"{prefix}_{suffix}{extension}"
        
        if output_dir:
            return os.path.join(output_dir, output_name)
        else:
            return os.path.join(os.path.dirname(input_filename), output_name)


class VolumeLoader:
    """Load and handle multi-volume image data."""
    
    @staticmethod
    def load_image_series(file_list: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load a series of images into a 4D volume.
        
        Args:
            file_list: List of image file paths
            
        Returns:
            Tuple of (4D_volume, metadata_dict)
        """
        if not file_list:
            raise ValueError("Empty file list provided")
        
        # Load first image to get dimensions
        first_data, first_header, first_affine = ImageLoader.load_nifti(file_list[0])
        
        # Initialize 4D volume
        volume_shape = list(first_data.shape) + [len(file_list)]
        volume = np.zeros(volume_shape, dtype=first_data.dtype)
        volume[..., 0] = first_data
        
        # Load remaining images
        for i, filename in enumerate(file_list[1:], 1):
            try:
                data, _, _ = ImageLoader.load_nifti(filename)
                
                # Check dimensions match
                if data.shape != first_data.shape:
                    logger.warning(f"Image {filename} has different dimensions, resizing...")
                    # Could implement resizing here if needed
                    continue
                
                volume[..., i] = data
                
            except Exception as e:
                logger.error(f"Could not load image {filename}: {e}")
                continue
        
        metadata = {
            'affine': first_affine,
            'header': first_header,
            'file_list': file_list,
            'shape': volume.shape
        }
        
        return volume, metadata
    
    @staticmethod
    def save_image_series(volume: np.ndarray, 
                         file_list: List[str],
                         output_dir: str,
                         prefix: str = "output") -> List[str]:
        """
        Save 4D volume as series of 3D images.
        
        Args:
            volume: 4D image volume
            file_list: Original file list (for metadata)
            output_dir: Output directory
            prefix: Filename prefix
            
        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []
        
        for i in range(volume.shape[-1]):
            output_filename = os.path.join(output_dir, f"{prefix}_{i:04d}.nii")
            ImageLoader.save_nifti(volume[..., i], output_filename)
            saved_files.append(output_filename)
        
        return saved_files