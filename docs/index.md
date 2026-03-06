# ROCKETSHIP

![ROCKETSHIP Banner](assets/rocketship_banner2.png)

ROCKETSHIP is a toolbox for processing and analyzing parametric MRI and DCE-MRI data. It was developed at the Biological Imaging Center at the California Institute of Technology and Loma Linda University.

## Citation

If you use ROCKETSHIP in your project, please cite:

Ng, T.S.C., et al. [ROCKETSHIP: a flexible and modular software tool for the planning, processing and analysis of dynamic MRI studies](https://doi.org/10.1186/s12880-015-0062-3). *BMC Medical Imaging*, 15, 19 (2015). PMID: 26076957

## Start Here

 
- [Python Walkthrough](wiki/python-walkthrough.md)
- [MATLAB, DCE Walkthrough](wiki/dce-walkthrough.md)
- [MATLAB, DSC Walkthrough](wiki/dsc-walkthrough.md)
- [MATLAB, Parametric Walkthrough](wiki/parametric-walkthrough.md)
- [Enable GPU Acceleration](wiki/enable-gpu-acceleration.md)
- [DCE Options Reference](dce_options.md)

## Python (Recommended) Quick Start

```bash
git clone https://github.com/petmri/ROCKETSHIP.git
cd ROCKETSHIP
python3 install_python_acceleration.py
source .venv/bin/activate
python run_dce_python_cli.py
python run_parametric_python_cli.py
```

For advanced Python usage and configuration details, see:
- [Python README (repository)](https://github.com/petmri/ROCKETSHIP/blob/dev/python/README.md)
- [Project Management Docs (repository)](https://github.com/petmri/ROCKETSHIP/blob/dev/docs/project-management/README.md)
- [Transition TODO (repository)](https://github.com/petmri/ROCKETSHIP/blob/dev/docs/project-management/TODO.md)
- [Python Roadmap (repository)](https://github.com/petmri/ROCKETSHIP/blob/dev/docs/project-management/ROADMAP.md)
- [Porting Status (repository)](https://github.com/petmri/ROCKETSHIP/blob/dev/docs/project-management/PORTING_STATUS.md)

## MATLAB (Legacy) Quick Start

1. Clone ROCKETSHIP: `git clone --recursive https://github.com/petmri/ROCKETSHIP.git`
2. Add the ROCKETSHIP folder to the MATLAB path
3. Calculate T1 maps with `run_parametric.m`
4. Check T1 maps with `run_analysis.m`
5. Calculate DCE maps with `run_dce.m`

## Selected Publications Using ROCKETSHIP

For a more complete list, see [Google Scholar](https://scholar.google.com/scholar?cites=17209875609254734596&as_sdt=2005&sciodt=0,5&hl=en).

- Pan, H., et al. [Liganded magnetic nanoparticles for magnetic resonance imaging of α-synuclein](https://doi.org/10.1038/s41531-025-00918-z). *npj Parkinson's Disease*, 11(1), 88 (2025). PMID: 40268938
- Llull, B., et al. [Blood-Brain Barrier Disruption Predicts Poor Outcome in Subarachnoid Hemorrhage: A Dynamic Contrast-Enhanced MRI Study](https://doi.org/10.1161/STROKEAHA.125.051455). *Stroke*, 56(9), 2633-2643 (2025). PMID: 40557536
- Reas, E.T., et al. [APOE ε4-related blood-brain barrier breakdown is associated with microstructural abnormalities](https://doi.org/10.1002/alz.14302). *Alzheimer's & Dementia*, 20(12), 8615-8624 (2024). PMID: 39411970
- Montagne, A., et al. [APOE4 leads to blood-brain barrier dysfunction predicting cognitive decline](https://pubmed.ncbi.nlm.nih.gov/32376954/). *Nature*, 581(7806), 71-76 (2020). PMID: 32376954
- Backhaus, P., et al. [Toward precise arterial input functions derived from DCE-MRI through a novel extracorporeal circulation approach in mice](https://pubmed.ncbi.nlm.nih.gov/32077523/). *Magnetic Resonance in Medicine*, 84(3), 1404-1415 (2020). PMID: 32077523
- Bagley, S.J., et al. [Clinical Utility of Plasma Cell-Free DNA in Adult Patients with Newly Diagnosed Glioblastoma: A Pilot Prospective Study](https://pubmed.ncbi.nlm.nih.gov/31666247/). *Clinical Cancer Research*, 26(2), 397-407 (2020). PMID: 31666247
- Ng, T.S.C., et al. [Clinical Implementation of a Free-Breathing, Motion-Robust Dynamic Contrast-Enhanced MRI Protocol to Evaluate Pleural Tumors](https://pubmed.ncbi.nlm.nih.gov/32348181/). *AJR American Journal of Roentgenology*, 215(1), 94-104 (2020). PMID: 32348181
- Pacia, C.P., et al. [Feasibility and safety of focused ultrasound-enabled liquid biopsy in the brain of a porcine model](https://pubmed.ncbi.nlm.nih.gov/32366915/). *Scientific Reports*, 10(1), 7449 (2020). PMID: 32366915
- Boehm-Sturm, P., et al. [Low-Molecular-Weight Iron Chelates May Be an Alternative to Gadolinium-based Contrast Agents for T1-weighted Contrast-enhanced MR Imaging](https://pubmed.ncbi.nlm.nih.gov/28880786/). *Radiology*, 286(2), 537-546 (2018). PMID: 28880786
- Sta Maria, N.S., et al. [Low Dose Focused Ultrasound Induces Enhanced Tumor Accumulation of Natural Killer Cells](https://doi.org/10.1371/journal.pone.0142767). *PLOS ONE*, 10(11), e0142767 (2015). PMID: 26556731

## Support

If you need help and cannot find it in the docs yet, contact Sam Barnes (`sabarnes@llu.edu`).
