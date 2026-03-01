# run_parametric.m and fitting_gui.m
run_parametric.m adds required paths to your Matlab path then runs fitting_gui.m which is the main script for parametric fitting of T1, T2/T2*, and ADC data. The parametric fitting is included in ROCKETSHIP as a [submodule](https://git-scm.com/docs/git-submodule), it is part of the [parametric_scripts](https://github.com/petmri/parametric_scripts) github project. If you are missing these files, when you clone ROCKETSHIP use the command: 

`git clone --recursive https://github.com/petmri/ROCKETSHIP.git` 

this will include the submodule. Alternatively you can clone the [parametric_scripts](https://github.com/petmri/parametric_scripts) project directly.

## Datasets to process
The parametric fitting has the option to queue up many jobs and run the sequentially. "Add Datasets" will add on additional jobs to be run after the first is finished. Each "Dataset" is a separate jobs and it processing will be configured individually.

## Email and CPUs
An email can be sent when all jobs have completed. For this option to work the file the file "email_preferences.txt" must be created an configured. For syntax look at the included file "email_preferences_example.txt". This file specifies a GMail account that the emails will be sent from, and file should be placed in the parametric_scripts folder.
* Email to alert: email address that will be alerted when job is finished
* Number CPUs: Specifies number of Matlab "workers" that will be used to complete the job. Recommended to set to number of cores available for processing.

## Job Information Logging
Specifies what data structures (*.mat files) and what logs to save
* Save text log?: save a *.txt file with the log output for every dataset job
* Email log?: will email log files to email specified above
* Save Data structures log?: Will save the *.mat file will all relevant variables after data processing is finished. Useful for saving the finished data in a format that makes additional processing easy.
* Save batch log?: Saves a log file for the entire batch run, relevant when running multiple dataset jobs
* Select old Data structure: selects a *.mat file from a configured "Dataset" to rerun.

## Files for Dataset #
Select files to processing, only NIFTI format is supported. "Up" "Down" buttons can be used to change file order if multiple files are required (e.g. one file for each FA). Only add multiple files here if they are part of a single fitting, to queue up multiple jobs use the "Add Datasets" option.

## Data Order and Other Parameters
* Data Order: selects what order the Z dimension and the parameter "N" dimension are in for 4D datasets. If the different parameters in seperate files (e.g. one file for each FA), select "X, Y, Z / File".
* TE/TR/FA/TI/B Value list: list the parameter values that were collected for the desired fitting, must be in the same order that they appear in the dataset. Units are ms for TE/TR/TI, degrees for FA, mm^2/s for B value.
* TR: For T1 multi FA and multi TI only, specify the TR used, in ms
* Only Odd Echoes: For T2/T2* fitting, only uses the odd echoes in the fitting, useful to correct for eddy currents if echoes were collected on positive and negative gradient lobes.
* Fit all voxels: fit a parametric value for each individual voxel in the image
* Output file basename: name of output file


## Fit Type
Specifies the type of fitting to use.
### T2
* T2 Linear Weighted
  - Linearize the decay by taking log(si), then performing a linear fit weighted by the original signal intensity. Faster than exponential, almost as accurate. 
  - Output T2/T2* in ms, sum of squared error, r^2 (coefficient of determination), and 95% confidence intervals.
* T2 Exponential
  - Fit using non-linear least squares with `A*exp(-N/B)` model
  - Output T2/T2* in ms, sum of squared error, r^2 (coefficient of determination), and 95% confidence intervals
* T2 Exp + C
  - Fit using non-linear least squares with `A*exp(-N/B)+C` model. Used for liver iron quantification, see [Wood et al.](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1895207/). To improve fitting stability, T2 is first estimated by looking at the ratio of the first and last echoes, if the estimate is <10ms the `A*exp(-N/B)+C` model is used, if the estimate is ≥10ms the `A*exp(-N/B)` model is used.
  - Output T2/T2* in ms, sum of squared error, r^2 (coefficient of determination), and 95% confidence intervals
* T2 Linear Fast
  - Linearize the decay by taking log(si), then performing a fast linear fit. Very fast.
  - Output T2/T2* in ms, and r^2 (coefficient of determination).

### T1
* T1 Multi TR:
  - Fit using non-linear least squares with `A*(1-exp(-N/B))` model
  - T1 restricted to 0 - 10000ms
  - Output T1 in ms, sum of squared error, r^2 (coefficient of determination), and 95% confidence intervals
* T1 Multi FA Exponential:
  - Fit using non-linear least squares with `A*((1-exp(-TR/B))*sin(N) )/( 1-exp(-TR/B)*cos(N) )` model
  - T1 restricted to 0 - 10000ms
  - Output T1 in ms, sum of squared error, r^2 (coefficient of determination), and 95% confidence intervals
* T1 Multi FA Linear:
  - Linearize and perform a fast linear fit using 
`y_lin = si./sin(pi/180*N);`
`x_lin = si./tan(pi/180*N);`
`T1   = -TR/log(slope);`
  - Output T1 in ms, and r^2 (coefficient of determination),
* T1 Multi TI:
  - Fit using non-linear least squares with `abs(A*(1-2*exp(-N/B)-exp(-TR/B)))` model
  - T1 restricted to 0 - 10000ms
  - No fast linear fit available, so no acceleration from r^2 estimate and threshold, recommended to trim noise areas
  - Output T1 in ms, sum of squared error, r^2 (coefficient of determination), and 95% confidence intervals

### ADC
* ADC Linear Weighted
  - Linearize the decay by taking log(si), then performing a linear fit weighted by the original signal intensity. Faster than exponential, almost as accurate. 
  - Output ADC in mm^2/s, sum of squared error, r^2 (coefficient of determination), and 95% confidence intervals.
* ADC Exponential
  - Fit using non-linear least squares with `A*exp(-N*B)` model
  - Output ADC in mm^2/s, sum of squared error, r^2 (coefficient of determination), and 95% confidence intervals
* ADC Linear Simple:
  - Linearize the decay by taking log(si), then performing a linear fit.
  - Output ADC in mm^2/s, sum of squared error, r^2 (coefficient of determination), and 95% confidence intervals
* ADC Linear Fast:
  - Linearize the decay by taking log(si), then performing a fast linear fit. Very fast.
  - Output ADC in mm^2/s, and r^2 (coefficient of determination).

## Estimated r^2 map
To speed up processing, an initial linear fit is performed an the r^2 (coefficient of determination) value is calculated. If the fit has a very low r^2 value, the more time intensive exponential fitting will not be performed. This can accelerate the processing without requiring that the background noise areas be manually removed.
* r^2 threshold: all values with an r^2 value lower than this value will not be fit, not applicable for "linear fast" options. If an estimated r^2 map has been calculated, voxels that will be kept with the current threshold are marked in red.
* Slice: slice of r^2 map that is currently being displayed
* Run/Update Estimate: calculates an r^2 map using the "linear fast" option, this is visualized above and is used to set the r^2 threshold properly