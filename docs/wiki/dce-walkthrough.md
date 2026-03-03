# Launch ROCKETSHIP
The main rocketship folder should be added to your matlab path. Then launch the DCE processing one of three ways:

\>>rocketship

This launches a GUI where you can selection DCE

\>>run_dce

This adds the required subfolders to your path the launches the DCE processing

\>>dce

This launches the DCE processing directly, only use if you have added the appropariate subfolders to your path. 

The DCE processing is broken into four steps: A, B, D, and E.

## Run A
This section loads all of the input images and ROI files, it then calculates the concentration vs. time curves. 

### Input dynamic datasets
Load the dynamic MRI images, usually a series of T1-weighted images. DICOM and NIFTI files are supported. "File Order" selects what order the "Z" dimension and time "t" dimension are.

### Select T1 and ROI files ...
* Select AIF/RR: defines what region to get the arterial input function (AIF) or reference region (RR) from the dynamic series. This can be a simple binary mask, with 1 in the AIF/RR region and zero elsewhere; or a T1 map, where the AIF/RR region contains valid T1 values (in ms), and all other regions are zero.
* Select ROI: defines the region where analysis can be performed, is used to mask out uninteresting anatomy and noise regions to save on processing time. This can be a simple binary mask, with 1 in the AIF/RR region and zero elsewhere; or a T1 map, where the AIF/RR region contains valid T1 values (in ms), and all other regions are zero.
* Select T1 map: if "Mask" was chosen for either of the above steps it is necessary to input a T1 map to define the T1 values for the tissues and AIF/RR. Units of ms, if in sec will attempt to auto detect and convert to ms.
* Select Drift ROI: optional, defines a region of the image to use for drift correction. This must have constant signal intensity over the entirety of the scan (i.e. a vial of water with Gd), any tissue will not work as there is always some uptake of contrast agent. Scanner drift correction will be applied. All signal intensities are normalized in time based on the visible phantom.
  * Global drift correction: Calculate a single drift correction factor to be applied to the entire image. Otherwise a different correction factor will be calculated and applied for each slice (if a slice doesn't have any visible drift phantom it will search for the nearest slice that does and use that).

Noise handling used for SNR calculation to exclude noisy AIF voxels from analysis. 
* Pick noise file: select binary mask image that defines a region of air that contains only noise.
* Derive noise from corner square: derive noise from top left region of image in a square of defined size.

### Image Parameters
* TR: dynamic series repetition time TR (in ms)
* FA: dynamic series flip angle (in degrees)
* Hematocrit: subject hematocit level, used for converting from blood concentrations to plasma concentrations
* SNR for AIF filter: all AIF voxels with SNR lower than this will be removed
* End Baseline, image num: the image number of the last baseline image before contrast arrival. if -1 will be asked to select from AIF plot, if -2 will be automatically determined
* Contrast agent r1: relaxivity on contrast agent used see [Shen et al.](http://www.ncbi.nlm.nih.gov/pubmed/25658049)
* Injection Duration (number images): how long the injection lasted in number of images, used for auto AIF selection

### AIF Selection
* ROI defined: AIF defined by user input ROI, T1 value taken from T1 map or AIF ROI file
* ROI w/ Static T1: AIF defined by user input ROI, T1 value defined here
* Auto: automatically finds voxels that look most like a typical AIF (fast rise, slow decay), T1 value taken from T1 map
* Auto: automatically finds voxels that look most like a typical AIF (fast rise, slow decay), T1 value defined here
* Blood T1: defines a static pre T1 value to use for the AIF, in ms

## Run B
This defines the timing parameters for analysis, and derives the AIF curve to be used. Requires output from Part A.

### Results of A
Select the *.mat file saved from Part A

### AIF
* Raw AIF: uses the raw datapoints
* Fitted AIF: fits the raw datapoints to a linear upslope and bi-exponential decay model, uses the fitted values for DCE calculation, can reduce noise, can reduce accuracy.
* Import AIF: import a saved AIF from a *.mat file. Can use results from a previously run Part B, if defined manually must contain the variables:
  - Cp_use - contains the AIF concentration dynamic curve in mM
  - Stlv_use - contains the AIF signal intensity dynamic curve in a.u., only used for AUC calculations
  - import_timer - contains the time vector specifying the time, in minutes, of each data point
  - import_start - specifies the end of the baseline / the start of the bolus arrival, in minutes
* Create average AIF: reads the AIF from multiple *.mat files (take from multiple subjects), and averages them together to create a single average AIF to use.
* Analysis Interval: restrict the DCE analysis to specified time interval, in min. For no restriction use 0 for start and end.
* Injection Duration: specify start and end of _BOLUS ARRIVAL_ in minutes. Use -1 for either field to automatically determine start/end (takes start as the end of steady state, end as the maximum data point). Used for AIF fitting only.
* Time resolution: temporal resolution of dynamic series, in seconds
* Manual time vector: load time vector from specified *.mat file. Useful for defining dynamic series with unequally spaced time points. *.mat file must contain variable "timer" with is the time vector, in minutes.

## Run D
There is no Part C (historical reasons). Part D calculates the DCE maps, it requires as input the results of Part B. 

### Results of B
Select the *.mat file saved from part B

### Select DCE models
For details on each model, formulas, and references see [Ng et al.](https://doi.org/10.1186/s12880-015-0062-3) Multiple models can be selected, an output file will be generated for each model.
* Tofts: Ktrans, Ve
* Patlak: Ktrans, Vp
* FXR: Shutter speed model see [Zhou et al.](http://www.ncbi.nlm.nih.gov/pubmed/15282806)
* Tofts w/ Vp: Ktrans, Ve, Vp
* Nested Model: Variable
* Tissue Uptake: Ktrans, Fp
* 2CXM: Ktrans, Ve, Vp, Fp
* Area under curve

### Smoothing
Smoothing can be performed in time, or in plane (XY directions). 
* Time smoothing is not typically recommended as the DCE fittings effectively smooths in the time dimension. Robust Local Regression time smoothing can be useful for removing outlier points in the time curve (from motion, artifact, etc.) to keep them from influencing the fit.
* XY smooth size: specify the standard deviation (in voxels) of a kernel for Gaussian smoothing in XY.

### Fitting
* ROIs to fit: specify an ROI to perform a single fit on. All voxels in the ROI will be averaged together at each time point, then a single DCE fit will be performed on the resulting time curve. ROI can be defined as an ImageJ ROI file (*.roi), or a NIFTI file, treated as a binary mask.
* Fit all voxels: perform a DCE fit on every voxel, time intensive.

### Number CPUs
Specify the number of workers to use in the DCE fitting. 0 will automatically set to max number allowed (number of available cores), -1 will set to max-1 (available cores - 1). -1 is useful for long processing when you still need to use the computer while processing.

## Run E
Runs the `fitting_analysis.m` script. This is used to analyze the time curves and DCE fits.

### Fitting Results
* Models to Analyze: Input the *.mat files that were saved from Part D
* Voxel Selection Image: for voxel analysis, image that will be displayed to allow selection of which voxel to analysis, usually one of the DCE parametric maps (such as Ktrans), or the T1 map.

### Fitting Analysis
* ROI List: List of ROI results, if no ROIs were processed in Part D this will be empty. Click on a ROI here to launch the results analysis.
* Show original unsmoothed data: show unsmoothed time curve if time smoothing was used.
* Show 95% confidence interval curves: shows 95% confidence interval fit curve
* Run Voxel Analysis: launches voxel analysis tool

### Statistical Model Comparison
* Run Akaike: compare models using Akaike information criteria see [Glatting et al.](http://www.ncbi.nlm.nih.gov/pubmed/18072493)
* Run F Test: compare models using F test see [Glatting et al.](http://www.ncbi.nlm.nih.gov/pubmed/18072493)
* Run FMI/FRI compare models using FMI/FRI see [Balvay et al.](http://www.ncbi.nlm.nih.gov/pubmed/16155897)
* Perform ROI Comparison: run stats on ROI results
* Perform Voxel Comparison: run stats on voxel results