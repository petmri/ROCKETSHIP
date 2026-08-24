# ROCKETSHIP

![ROCKETSHIP Banner](assets/rocketship_banner2.png)

ROCKETSHIP is an open source toolbox for the processing and analysis of dynamic
contrast-enhanced (DCE) and parametric MRI data. It provides quantitative pharmacokinetic
modelling, parametric mapping of T<sub>1</sub>, T<sub>2</sub>, T<sub>2</sub><sup>*</sup> and
the apparent diffusion coefficient, and the supporting workflow required to take a study from
acquired images to parameter maps.

It was developed at the Biological Imaging Center at the California Institute of Technology
and at Loma Linda University, and is used in clinical and preclinical imaging research
worldwide.

## DCEasy

ROCKETSHIP is part of the [**DCEasy** family of software](https://petmri.github.io) for quantitative dynamic
contrast-enhanced MRI, 
DCEasy provides an integrated set of tools covering the full path from scanner output to
quantitative results:

- **DICOM to BIDS conversion**, preparing acquired studies into a standard organised layout
  with the acquisition metadata that quantitative analysis requires.
- **Automatic and manual arterial input function selection**, addressing the largest single
  source of systematic error in quantitative DCE-MRI.
- **Batch processing**, applying a validated analysis consistently across entire studies
  rather than one session at a time.
- **Pharmacokinetic modelling and parametric mapping**, provided by ROCKETSHIP itself.

The components are designed to work together, so a study prepared with the DCEasy conversion
and input function tools can be processed by ROCKETSHIP without further preparation. Each is
also usable independently. See [petmri.github.io](https://petmri.github.io) for a complete list of 
available tools.


## Citation

If you use ROCKETSHIP in your work, please cite:

Ng, T.S.C., et al. [ROCKETSHIP: a flexible and modular software tool for the planning,
processing and analysis of dynamic MRI studies](https://doi.org/10.1186/s12880-015-0062-3).
*BMC Medical Imaging*, 15, 19 (2015). PMID: 26076957

## Start here

- [Python Walkthrough](wiki/python-walkthrough.md) — the recommended interface for new work
- [MATLAB DCE Walkthrough](wiki/dce-walkthrough.md)
- [MATLAB DSC Walkthrough](wiki/dsc-walkthrough.md)
- [MATLAB Parametric Walkthrough](wiki/parametric-walkthrough.md)
- [GPU and CPU Acceleration](wiki/enable-gpu-acceleration.md)

Technical reference:

- [DCE Options](dce_options.md) — every configuration option
- [Signal to Concentration](reference/signal-to-concentration.md) — the conversion math and its assumptions
- [Pharmacokinetic Models](reference/models/index.md) — equations, parameters and model selection

## Python quick start

The Python implementation is the recommended interface and is under active development.

```bash
git clone https://github.com/petmri/ROCKETSHIP.git
```

```bash
cd ROCKETSHIP && python3 install.py
```

Launch the GUI:

```bash
./rocketship_dce.sh
```

Or work from the command line:

```bash
source .venv/bin/activate
```

```bash
python run_parametric_python_cli.py
```

```bash
python run_dce_python_cli.py
```

Parametric T<sub>1</sub> mapping comes first, since DCE analysis requires a pre-contrast T<sub>1</sub> map
as input. Graphical interfaces are available for both steps through
`run_dce_python_gui.py` and `run_parametric_python_gui.py`.

See the [Python Walkthrough](wiki/python-walkthrough.md) for the full procedure.

## MATLAB quick start (legacy)

The MATLAB implementation is the original version and will be maintained, but new features will only be 
implemented in python.

```bash
git clone https://github.com/petmri/ROCKETSHIP.git
```

1. Add the ROCKETSHIP folder to your MATLAB path.
2. Calculate T<sub>1</sub> maps with `run_parametric.m`.
3. Check the T<sub>1</sub> maps with `run_analysis.m`.
4. Calculate DCE maps with `run_dce.m`.

## Selected publications

A more complete list is available on
[Google Scholar](https://scholar.google.com/scholar?cites=17209875609254734596&as_sdt=2005&sciodt=0,5&hl=en).

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

Questions not answered by this documentation may be directed to Sam Barnes at
`sabarnes@llu.edu`. Bug reports and feature requests are best raised as
[issues on GitHub](https://github.com/petmri/ROCKETSHIP/issues).

## License

ROCKETSHIP is released under the terms described in the
[LICENSE](https://github.com/petmri/ROCKETSHIP/blob/master/LICENSE) file.
