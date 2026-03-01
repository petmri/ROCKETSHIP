# ROCKETSHIP v1.2,  2016
---
_Copyright (c) 2016, Thomas Ng, Samuel Barnes_

All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Contains program and associated files to process and analyze parametric MRI and DCE-MRI files. Developed at the Biological Imaging Center at the California Institute of Technology. 

### [SEE GITHUB WIKI FOR HELP](https://github.com/petmri/ROCKETSHIP/wiki)

Please see Barnes, Ng et al BMC Bioinformatics (2015) for more information.

Thomas Ng 	thomasn@caltech.edu

Samuel Barnes 	srbarnes@caltech.edu



## ROCKETSHIP Papers

If you use ROCKETSHIP in your project please reference

[Ng et al. ROCKETSHIP: a flexible and modular software tool for the planning, processing and analysis of dynamic MRI studies. BMC Med Img 2015](https://doi.org/10.1186/s12880-015-0062-3)

in any papers. This reference also has detailed information about the various DCE models used in this project. If you are pursing BBB human applications please consider these papers for parameter suggestions

[Barnes et al. Optimal acquisition and modeling parameters for accurate assessment of low Ktrans blood-brain barrier permeability using dynamic contrast-enhanced MRI. MRM 2015](http://www.ncbi.nlm.nih.gov/pubmed/26077645)

[Montagne et al. Blood-brain barrier breakdown in the aging human hippocampus. Neuron 2015](http://www.ncbi.nlm.nih.gov/pubmed/25611508)

or this for mouse applications

[Montagne et al. Pericyte degeneration causes white matter dysfunction in the mouse central nervous system. Nature Medicine 2018](https://pubmed.ncbi.nlm.nih.gov/29400711/)

Other Publications using ROCKETSHIP for a more complete list see [google scholar](https://scholar.google.com/scholar?cites=17209875609254734596&as_sdt=2005&sciodt=0,5&hl=en)
* [Sta Maria et al. Low Dose Focused Ultrasound Induces Enhanced Tumor Accumulation of Natural Killer Cells. PLOS One 2015](http://doi.org/10.1371/journal.pone.0142767)
* [Montagne et al. APOE4 leads to blood–brain barrier dysfunction predicting cognitive decline. Nature 2020](https://pubmed.ncbi.nlm.nih.gov/32376954/)
* [Backhaus et al. Toward precise arterial input functions derived from DCE‐MRI through a novel extracorporeal circulation approach in mice. MRM 2020](https://pubmed.ncbi.nlm.nih.gov/32077523/)
* [Bagley et al. Clinical Utility of Plasma Cell-Free DNA in Adult Patients with Newly Diagnosed Glioblastoma: A Pilot Prospective Study. Clinical Cancer Research 2020](https://pubmed.ncbi.nlm.nih.gov/31666247/)
* [Ng et al. Clinical Implementation of a Free-Breathing, Motion-Robust Dynamic Contrast-Enhanced MRI Protocol to Evaluate Pleural Tumors. American Journal of Roentgenology 2020](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7341067/)
* [Pacia et al. Feasibility and safety of focused ultrasound-enabled liquid biopsy in the brain of a porcine model. Scientific Reports 2020](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7198482/)
* [Boehm-Sturm et al. Low-Molecular-Weight Iron Chelates May Be an Alternative to Gadolinium-based Contrast Agents for T1-weighted Contrast-enhanced MR Imaging. Radiology 2017](https://pubmed.ncbi.nlm.nih.gov/28880786/) 
## ROCKETSHIP Requirements:

* Matlab Version
  * Verified Working: Matlab 2014a, 2014b
  * Should Work: Matlab 2015a, 2015b (please let me know if you have tried this)
  * Will Not Work: Matlab 2011
* Toolboxes:
  * Curve fitting
  * Parallel
  * Statistics
  * Image processing
  * Optimization (currently required for some functions, working to remove this requirement)
* Computer:
  * Some of the processing is very CPU intensive, a modern multi-core (≥4) processor, while not required, helps keep the processing time reasonable (heavily dependent on image matrix size).
  * (Optional) An NVIDIA GPU can be used to significantly speed up processing by using the [gpufit](https://github.com/ironictoo/Gpufit) library.

## File formats

ROCKETSHIP prefers all images to be input in the NIFTI format. DCE fitting does have some limited support for directly processing DICOM images, but it is recommended to convert from DICOM to NIFTI first, then use the NIFTI images for all processing. Additionally the parametric fitting (T1, T2, ADC) requires NIFTI files. To convert from DICOM to NIFTI we recommend using the dcm2nii tool that comes with MRIcron, it is available for Windows, Linux, and Mac and is easy to use (although any converter can be used). For dcm2nii select the FSL 4D NIFTI format. Compressed NIFTI images (.nii.gz) can be read by ROCKETSHIP, but not written.

## Quick Start

1. Clone ROCKETSHIP git clone --recursive https://github.com/petmri/ROCKETSHIP.git
2. Add ROCKETSHIP folder to Matlab path
3. Calculate T1 maps with script run_parametric.m
4. Check T1 maps with script run_analysis.m
5. Calculate DCE maps with script run_dce.m

## Additional Help

If you need help and can't find it here please contact Sam Barnes sabarnes@llu.edu.



## Listing of files and brief descriptor

### README.md				- This file

### i) dce: ROCKETSHIP DCE module  (94 files)
```
'ADDLUT.m'				- File parsing helper file		
'AIFbiexpcon.m'				- Fits AIF to biexponential model
'AIFbiexpfithelp.m'			- Helping file for AIF fitting
'A_make_R1maps_func.m'			- Submodule A for processing DCE-MRI files
'B_AIF_fitting_func.m'			- Submodule B for processing AIF/Reference regions
'D_fit_voxels_batch_func.m'		- Submodule D for curve fitting batch program
'D_fit_voxels_func.m'			- Submodule D for curve fitting
'FXLfit_generic.m'			- Model fitting file
'REMOVELUT.m'				- File parsing helper file 
'RUNA.fig'				- GUI for submodule A
'RUNA.m'				- GUI for submodule A
'RUNB.fig'				- GUI for submodule B
'RUNB.m'				- GUI for submodule B
'RUND.fig'				- GUI for submodule D
'RUND.m'				- GUI for submodule D
'allnumeric.m'				- File parsing helper file 
'appendLISTS.m'				- File parsing helper file 
'auc_helper.m'				- File for area-under curve calculation
'average_aifs.fig'			- GUI for submodule B: generate population AIF
'average_aifs.m'			- GUI for submodule B: generate population AIF
'cleanAB.m'				- Helper file for submodule A
'cleanR1t.m'				- Helper file for submodule A
'compare_fits.m'			- Helper file for submodule E
'compare_gui.fig'			- Helper file for submodule E
'compare_gui.m'				- Helper file for submodule E
'consistencyCHECKRUNA.m'		- Helper file for submodule A
'dce.fig'				- MAIN File for ROCKETSHIP module GUI
'dce.m'					- MAIN File for ROCKETSHIP module GUI
'dce_auto_aif.m'			- Helper file for submodule A
'dce_last_state.mat'			- Data file storing previous GUI states
'dce_preferences.txt'			- Preference file containing parameters for fitting. Edit as necessary
'disp_error.m'				- Helper file to display error messages in GUIs
'double2uint16Scale.m'			- Helper file to parse DICOM images
'findLUTPLACE.m'			- File parsing helper file 
'findRod.m'				- Helper file for submodule A
'findrootname.m'			- File parsing helper file 
'findsubjectID.m'			- File parsing helper file
'finduniqueID.m'			- File parsing helper file
'finduniqueID_B.m'			- File parsing helper file
'finduniqueID_B_helper.m'		- File parsing helper file
'finduniqueIDhelper.m'			- File parsing helper file
'finduniqueIDhelperB.m'			- File parsing helper file
'fitting_analysis.fig'			- GUI for submodule E
'fitting_analysis.m'			- GUI for submodule E
'generatefullpath.m'			- File parsing helper file
'isDICOM.m'				- File parsing helper file
'isDICOMhdr.m'				- File parsing helper file
'isNIFTI.m'				- File parsing helper file
'license.txt'				- License file
'loadIMGVOL.m'				- File parsing helper file
'longestfilename.m'			- File parsing helper file
'matchindex.m'				- File parsing helper file
'model_0.m'				- Helper file for submodule D: model fitting
'model_2cxm.m'				- Helper file for submodule D: model fitting
'model_2cxm_cfit.m'			- Helper file for submodule D: model fitting
'model_extended_tofts.m'		- Helper file for submodule D: model fitting
'model_extended_tofts_cfit.m'		- Helper file for submodule D: model fitting
'model_fxr.m'				- Helper file for submodule D: model fitting
'model_fxr_cfit.m'			- Helper file for submodule D: model fitting
'model_patlak.m'			- Helper file for submodule D: model fitting
'model_patlak_cfit.m'			- Helper file for submodule D: model fitting
'model_patlak_linear.m'			- Helper file for submodule D: model fitting
'model_patlak_linear_fit.m'		- Helper file for submodule D: model fitting
'model_tissue_uptake.m'			- Helper file for submodule D: model fitting
'model_tissue_uptake_cfit.m'		- Helper file for submodule D: model fitting
'model_tofts.m'				- Helper file for submodule D: model fitting
'model_tofts_cfit.m'			- Helper file for submodule D: model fitting
'model_vp.m'				- Helper file for submodule D: model fitting
'model_vp_cfit.m'			- Helper file for submodule D: model fitting
'myginput.m'				- Helper file for submodule D: model fitting
'natORDER.m'				- File parsing helper file: sorts lists in natural order
'natORDERhelper.m'			- File parsing helper file: sorts lists in natural order
'nested_fit_helper.m'			- Helper file for submodule D: model fitting
'niftipathfind.m'			- File parsing helper file: sorts lists in natural order
'parse_cfit.m'				- Helper file for submodule D
'parse_cfit_helper.m'			- Helper file for submodule D
'parse_preference_file.m'		- Parses preference text file as needed.
'plot_curve.m'				- Plot time activity curves
'plot_dce_curve.m'			- Plot time activity curves
'reorderLUT.m'				- File parsing helper file
'rescaleDICOM.m'			- Helper file to parse DICOM images
'run_neuroecon_job.m'			- Legacy file to enable batch job on external server
'sortIMGVOL.m'				- Image parsing helper file
'sort_parse_2Dvol.m'			- Image parsing helper file
'sort_parse_2Dvol_helper.m'		- Image parsing helper file
'sort_parse_3Dvol.m'			- Image parsing helper file
'sort_parse_3Dvol_helper.m'		- Image parsing helper file
'sort_parse_INPUTS.m'			- Image parsing helper file
'split2subsets.m'			- Image parsing helper file
'updateLUT.m'				- File parsing helper file
'update_segmentlist.m'			- File parsing helper file
'visualize_list.m'			- File parsing helper file
'visualize_list_dce.m'			- File parsing helper file
'visualize_runD.m'			- File parsing helper file
```

### ii) parametric_scripts 
ROCKETSHIP fitting module to generate T2/T2*, T1, ADC and other parametric maps (31 files)
```
'calculateMap.m'			- Helper file for fitting
'calculateMap_batch.m'			- Helper file for fitting
'check_TRfit.m'				- Helper file to check TR fitting
'disp_error.m'				- Helper file to display error messages in GUI
'double2uint16Scale.m'			- Image parsing helper file 
'fitParameter.m'			- Helper file for fitting
'fitting_gui.fig'			- MAIN file for ROCKETSHIP fitting module
'fitting_gui.m'				- MAIN file for ROCKETSHIP fitting module
'fitting_script.m'			- Helper file for fitting
'generatefullpath.m'			- File parsing helper file
'instantiate_dataset.m'			- File parsing helper file
'isDICOM.m'				- File parsing helper file
'license.txt'				- License file
'load_batch.m'				- File parsing helper file
'load_image_files.m'			- File parsing helper file
'makeNewbatch.m'			- File parsing helper file
'miscellaneous_aux_files'		- Directory containing miscellaneous auxilary files, likely unnecessary, but kept for legacy purposes.
'parallelFit.m'				- Helper file for fitting
'parseDICOM.m'				- File parsing helper file
'prepareFit.m'				- Helper file for fitting
'preprocessIMGvol.m'			- Image parsing helper file
'quick_check.m'				- Check for error prior to fitting
'rescaleDICOM.m'			- Image parsing helper file
'setup_file_list.m'			- File parsing helper file
'setup_job.m'				- Helper file for fitting
'sortIMGVOL.m'				- Image parsing helper file
'update_handles.m'			- Helper file for fitting
'update_parameters.m'			- Helper file for fitting
'visualize_R2.m'			- Helper file for GUI, visualize R2 maps
'visualize_list.m'			- Image parsing helper file
```

### iii) external_programs

Listing of various programs incorporated under BSD license as part of ROCKETSHIP. Please go to the relevant page on MATLABCentral for details for each particular app.
```
'DataHash'
'ProgressBar'
'ReadImageJROI'
'TimedProgressBar'
'dirr'
'ftest'
'imshow3d'
'niftitools'
'uirecall'
```

