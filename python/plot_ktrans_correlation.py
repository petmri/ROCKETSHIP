#!/usr/bin/env python3
"""Compare two images voxel-by-voxel with correlation statistics.

Examples:
    python plot_ktrans_correlation.py map1.nii.gz map2.nii.gz
    python plot_ktrans_correlation.py t1_ref.nii.gz t1_test.nii.gz --quantity "T1 (ms)"
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import nibabel as nib


def load_nifti_data(filepath):
    """Load NIfTI file and return data array."""
    img = nib.load(filepath)
    return img.get_fdata()


def compute_correlation_stats(x, y):
    """Compute correlation statistics between two arrays."""
    # Pearson correlation
    r_pearson, p_pearson = stats.pearsonr(x, y)
    
    # Spearman correlation
    r_spearman, p_spearman = stats.spearmanr(x, y)
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # RMSE
    rmse = np.sqrt(np.mean((y - x) ** 2))
    
    # Mean absolute error
    mae = np.mean(np.abs(y - x))
    
    # Mean values
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    return {
        'r_pearson': r_pearson,
        'p_pearson': p_pearson,
        'r_spearman': r_spearman,
        'p_spearman': p_spearman,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'rmse': rmse,
        'mae': mae,
        'mean_x': mean_x,
        'mean_y': mean_y,
        'ratio_mean': mean_y / mean_x if mean_x != 0 else np.nan,
    }


def _auto_axis_limits(x, y, lower_pct=1.0, upper_pct=99.0):
    """Choose robust scatter-plot limits from central percentiles."""
    stacked = np.concatenate([x, y])
    lo = float(np.percentile(stacked, lower_pct))
    hi = float(np.percentile(stacked, upper_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo = float(np.min(stacked))
        hi = float(np.max(stacked))
    if lo >= hi:
        hi = lo + 1.0
    margin = 0.03 * (hi - lo)
    return lo - margin, hi + margin


def plot_correlation(
    x,
    y,
    stats_dict,
    label_x='Map 1',
    label_y='Map 2',
    quantity='Value',
    axis_min=None,
    axis_max=None,
    output_path=None,
):
    """Create scatter plot with unity line and regression line."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    ax1.scatter(x, y, alpha=0.3, s=1, c='blue', rasterized=True)
    
    # Unity line
    lim_min = min(np.min(x), np.min(y))
    lim_max = max(np.max(x), np.max(y))
    ax1.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5, linewidth=1, label='Unity line')
    
    # Regression line
    x_line = np.array([lim_min, lim_max])
    y_line = stats_dict['slope'] * x_line + stats_dict['intercept']
    ax1.plot(x_line, y_line, 'r-', alpha=0.7, linewidth=2, label='Linear fit')
    
    ax1.set_xlabel(f'{label_x} {quantity}', fontsize=12)
    ax1.set_ylabel(f'{label_y} {quantity}', fontsize=12)
    ax1.set_title('Voxel-by-Voxel Correlation', fontsize=14, fontweight='bold')
    if axis_min is None or axis_max is None:
        lim_min, lim_max = _auto_axis_limits(x, y)
    else:
        lim_min, lim_max = float(axis_min), float(axis_max)
    ax1.set_xlim(lim_min, lim_max)
    ax1.set_ylim(lim_min, lim_max)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = (
        f"N voxels = {len(x):,}\n"
        f"Pearson r = {stats_dict['r_pearson']:.4f} (p={stats_dict['p_pearson']:.2e})\n"
        f"R² = {stats_dict['r_squared']:.4f}\n"
        f"Slope = {stats_dict['slope']:.4f}\n"
        f"Intercept = {stats_dict['intercept']:.4e}\n"
        f"Mean ratio = {stats_dict['ratio_mean']:.4f}\n"
        f"RMSE = {stats_dict['rmse']:.4e}\n"
        f"MAE = {stats_dict['mae']:.4e}"
    )
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Bland-Altman plot
    mean_vals = (x + y) / 2
    diff_vals = y - x
    mean_diff = np.mean(diff_vals)
    std_diff = np.std(diff_vals)
    
    ax2.scatter(mean_vals, diff_vals, alpha=0.3, s=1, c='green', rasterized=True)
    ax2.axhline(mean_diff, color='red', linestyle='-', linewidth=2, label=f'Mean diff = {mean_diff:.4e}')
    ax2.axhline(mean_diff + 1.96 * std_diff, color='red', linestyle='--', linewidth=1, 
                label=f'+1.96 SD = {mean_diff + 1.96 * std_diff:.4e}')
    ax2.axhline(mean_diff - 1.96 * std_diff, color='red', linestyle='--', linewidth=1,
                label=f'-1.96 SD = {mean_diff - 1.96 * std_diff:.4e}')
    ax2.axhline(0, color='black', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel(f'Mean {quantity}', fontsize=12)
    ax2.set_ylabel(f'{label_y} - {label_x}', fontsize=12)
    ax2.set_title('Bland-Altman Plot', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
                description='Compare two images with voxel-by-voxel correlation plot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_ktrans_correlation.py matlab_ktrans.nii.gz python_ktrans.nii.gz
    python plot_ktrans_correlation.py map1.nii.gz map2.nii.gz --mask roi.nii.gz --output correlation.png --quantity "Ktrans"
    python plot_ktrans_correlation.py t1_ref.nii.gz t1_test.nii.gz --quantity "T1 (ms)" --threshold 1.0 --axis-max 3000
        """
    )
    
    parser.add_argument('map1', type=str, help='First map (NIfTI file)')
    parser.add_argument('map2', type=str, help='Second map (NIfTI file)')
    parser.add_argument('--mask', type=str, default=None, help='Optional mask (NIfTI file)')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output plot filename (default: display interactive plot)')
    parser.add_argument('--threshold', type=float, default=0.0, help='Minimum map threshold to include voxels (default: 0.0)')
    parser.add_argument('--label1', type=str, default='Map 1', help='Label for first map (default: Map 1)')
    parser.add_argument('--label2', type=str, default='Map 2', help='Label for second map (default: Map 2)')
    parser.add_argument('--quantity', type=str, default='Value', help='Physical quantity label used in axes (e.g., "T1 (ms)", "Ktrans")')
    parser.add_argument('--axis-min', type=float, default=None, help='Optional fixed scatter axis lower limit (applies to X and Y).')
    parser.add_argument('--axis-max', type=float, default=None, help='Optional fixed scatter axis upper limit (applies to X and Y).')
    parser.add_argument('--exclude-value', type=float, action='append', default=[-1.0], help='Exclude voxels where either map equals this value (repeatable). Default excludes -1.0 sentinel.')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading {args.map1}...")
    map1_data = load_nifti_data(args.map1)
    
    print(f"Loading {args.map2}...")
    map2_data = load_nifti_data(args.map2)
    
    # Check shapes match
    if map1_data.shape != map2_data.shape:
        print(f"ERROR: Map shapes don't match: {map1_data.shape} vs {map2_data.shape}")
        sys.exit(1)
    
    # Load mask if provided
    if args.mask:
        print(f"Loading mask {args.mask}...")
        mask_data = load_nifti_data(args.mask)
        if mask_data.shape != map1_data.shape:
            print(f"ERROR: Mask shape doesn't match: {mask_data.shape} vs {map1_data.shape}")
            sys.exit(1)
        mask = mask_data > 0
    else:
        mask = np.ones(map1_data.shape, dtype=bool)
    
    # Apply threshold (exclude near-zero and negative values)
    valid_mask = mask & np.isfinite(map1_data) & np.isfinite(map2_data)

    for value in args.exclude_value:
        valid_mask &= ~np.isclose(map1_data, value) & ~np.isclose(map2_data, value)

    if args.threshold > 0:
        valid_mask &= (map1_data >= args.threshold) | (map2_data >= args.threshold)
    
    # Extract valid voxels
    x = map1_data[valid_mask].flatten()
    y = map2_data[valid_mask].flatten()
    
    print(f"\nTotal voxels in volume: {map1_data.size:,}")
    print(f"Valid voxels (after masking/thresholding): {len(x):,}")
    print(f"Map 1 range: [{np.min(x):.6e}, {np.max(x):.6e}]")
    print(f"Map 2 range: [{np.min(y):.6e}, {np.max(y):.6e}]")
    
    if len(x) == 0:
        print("ERROR: No valid voxels found!")
        sys.exit(1)
    
    # Compute statistics
    print("\nComputing correlation statistics...")
    stats_dict = compute_correlation_stats(x, y)
    
    # Print statistics
    print("\n" + "="*60)
    print("CORRELATION STATISTICS")
    print("="*60)
    print(f"Number of voxels:        {len(x):,}")
    print(f"Pearson correlation:     r = {stats_dict['r_pearson']:.6f}, p = {stats_dict['p_pearson']:.2e}")
    print(f"Spearman correlation:    ρ = {stats_dict['r_spearman']:.6f}, p = {stats_dict['p_spearman']:.2e}")
    print(f"R-squared:               {stats_dict['r_squared']:.6f}")
    print(f"Linear fit:              y = {stats_dict['slope']:.6f}*x + {stats_dict['intercept']:.6e}")
    print(f"Mean {args.label1}:      {stats_dict['mean_x']:.6e}")
    print(f"Mean {args.label2}:      {stats_dict['mean_y']:.6e}")
    print(f"Mean ratio (y/x):        {stats_dict['ratio_mean']:.6f}")
    print(f"RMSE:                    {stats_dict['rmse']:.6e}")
    print(f"MAE:                     {stats_dict['mae']:.6e}")
    print("="*60)
    
    # Create plot
    print("\nGenerating plots...")
    plot_correlation(
        x,
        y,
        stats_dict,
        label_x=args.label1,
        label_y=args.label2,
        quantity=args.quantity,
        axis_min=args.axis_min,
        axis_max=args.axis_max,
        output_path=args.output,
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
