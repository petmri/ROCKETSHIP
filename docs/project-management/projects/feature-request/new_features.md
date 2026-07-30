# New Feature Requests
-[ ] Turn the noise estimation currently used in pipeline parity testing into a standalone function for general use. This is a small but useful utility that can be used to estimate noise in DCE-MRI voxels, and could be useful for other applications as well (AIF voxel filtering, TV baseline detection, etc.).
- [ ] Implement confidence intervals for GPUfit/CPUfit (currently only available in MATLAB/Python).
- [ ] Add parallelization to CPUfit (OpenMP?)
- [ ] Auto detect when to chop, look at baseline and determine if it is "flat enough" to use the full baseline, or if chopping is required. If chopping is required, determine how many frames to chop. Could also look at osciliations in the z direction to determine if chopping is required.