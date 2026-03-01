# Launch ROCKETSHIP
The main rocketship folder should be added to your matlab path. Then launch the DCE processing one of three ways:

\>>rocketship

This launches a GUI where you can selection DCE

\>>run_dsc

This adds the required subfolders to your path the launches the DSC processing

\>>dsc

This launches the DSC processing directly, only use if you have added the appropriate subfolders to your path. 

# DSC Processing
### Image Selection
Select the source DSC images, these must be in NIFTI file format.
### Noise Handling
Select an area of the image containing only air that is used to estimate the noise level of the image. This can be automatically selected in the corner of the image, or by a user specified binary mask.
### Define AIF
The AIF can be defined in three ways.
1. "User Selected AIF" - The AIF is defined by a user specified binary mask (NIFTI file)
2. "Import AIF" - The AIF is defined in a Matlab .mat file
3. "Use AIF from Previous Run" - The AIF is imported from the last ROCKETSHIP run
### Fitting Function
Define the type of fitting function used
1. Biexponential
2. Biexponential local
3. Gamma-Variant
4. Raw Data - No fitting
5. Upslope copy biexponential linear adjustment
6. Upslope copy biexponential
### Input Parameters and Deconvolution Algorithm
Specify the scan parameters, tissue parameters, and algorithm to use
### Bolus injection time
1. Automatic - Bolus arrival time will be automatically detected from the signal intensity curves
2. User Selected - User will be prompted to identify bolus arrival time

