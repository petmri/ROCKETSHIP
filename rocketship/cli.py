"""
Command Line Interface for ROCKETSHIP

Provides CLI entry points for DCE, parametric, and analysis functions.
"""

import click
import sys
import os

@click.group()
@click.version_option(version="2.0.0")
def cli():
    """ROCKETSHIP: Processing and analysis of dynamic MRI studies."""
    pass

@cli.command()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--input-dir', '-i', help='Input directory containing NIFTI files')
@click.option('--output-dir', '-o', help='Output directory for results')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def dce(config, input_dir, output_dir, verbose):
    """Run DCE-MRI analysis."""
    from .dce.main import run_dce_analysis
    
    if verbose:
        click.echo("Starting DCE-MRI analysis...")
    
    try:
        run_dce_analysis(
            config_file=config,
            input_dir=input_dir,
            output_dir=output_dir,
            verbose=verbose
        )
        click.echo("DCE analysis completed successfully.")
    except Exception as e:
        click.echo(f"Error during DCE analysis: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--input-files', '-i', multiple=True, help='Input NIFTI files')
@click.option('--parameters', '-p', multiple=True, type=float, help='Parameters (TE, TR, etc.)')
@click.option('--output-dir', '-o', help='Output directory for results')
@click.option('--fit-type', '-t', default='T2', help='Fit type (T1, T2, T2star, ADC)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def parametric(config, input_files, parameters, output_dir, fit_type, verbose):
    """Run parametric mapping (T1, T2, ADC, etc.)."""
    from .parametric.main import run_parametric_fitting
    
    if verbose:
        click.echo(f"Starting {fit_type} parametric fitting...")
    
    try:
        run_parametric_fitting(
            config_file=config,
            input_files=list(input_files),
            parameters=list(parameters),
            output_dir=output_dir,
            fit_type=fit_type,
            verbose=verbose
        )
        click.echo("Parametric fitting completed successfully.")
    except Exception as e:
        click.echo(f"Error during parametric fitting: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--input-dir', '-i', help='Input directory containing results')
@click.option('--output-dir', '-o', help='Output directory for analysis')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def analysis(input_dir, output_dir, verbose):
    """Run analysis and visualization of results."""
    from .utils.analysis import run_analysis
    
    if verbose:
        click.echo("Starting result analysis...")
    
    try:
        run_analysis(
            input_dir=input_dir,
            output_dir=output_dir,
            verbose=verbose
        )
        click.echo("Analysis completed successfully.")
    except Exception as e:
        click.echo(f"Error during analysis: {e}", err=True)
        sys.exit(1)

# Legacy entry points for backward compatibility
def dce_main():
    """Entry point for rocketship-dce command."""
    cli.main(['dce'] + sys.argv[1:], standalone_mode=False)

def parametric_main():
    """Entry point for rocketship-parametric command."""
    cli.main(['parametric'] + sys.argv[1:], standalone_mode=False)

def analysis_main():
    """Entry point for rocketship-analysis command."""
    cli.main(['analysis'] + sys.argv[1:], standalone_mode=False)

if __name__ == '__main__':
    cli()