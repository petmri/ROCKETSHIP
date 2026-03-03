# OSIPI Accuracy Summary

Computed from ROCKETSHIP fits against imported OSIPI datasets and compared to OSIPI posted peer-result aggregates.

- ROCKETSHIP datasets: `tests/data/osipi/...`
- Peer reference summary: `tests/data/osipi/reference/osipi_peer_error_summary.json`
- Peer source: https://github.com/OSIPI/DCE-DSC-MRI_TestResults (commit `23d3714797045d8103d5b5fa4f4c016840094dc0`)
- Figures:
  - `tests/data/osipi/reference/figures/osipi_accuracy_dros.png`
  - `tests/data/osipi/reference/figures/osipi_accuracy_patlak_delay.png`
  - `tests/data/osipi/reference/figures/osipi_accuracy_t1.png`

| Model | Dataset slice | Param | N | Our MAE | Our P95 | Our Max | Peer MAE | Peer P95 | Peer Max | MAE Ratio (Our/Peer) | Max Ratio (Our/Peer) | Within Peer Max | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- |
| tofts | OSIPI Tofts DRO | Ktrans | 25 | 0.000791678 | 0.00174196 | 0.00223753 | 0.000965054 | 0.00248998 | 0.00368751 | 0.820 | 0.607 | yes |  |
| tofts | OSIPI Tofts DRO | ve | 25 | 0.000722405 | 0.00190286 | 0.00424664 | 0.000849338 | 0.00211409 | 0.0042476 | 0.851 | 1.000 | yes |  |
| etofts | OSIPI Extended Tofts DRO | Ktrans | 15 | 0.000307117 | 0.00128792 | 0.00183079 | 0.000367743 | 0.00156536 | 0.00351224 | 0.835 | 0.521 | yes |  |
| etofts | OSIPI Extended Tofts DRO | ve | 15 | 0.000620012 | 0.00163846 | 0.00197435 | 0.00113827 | 0.00396572 | 0.00751087 | 0.545 | 0.263 | yes |  |
| etofts | OSIPI Extended Tofts DRO | vp | 15 | 0.000270511 | 0.000956455 | 0.00130541 | 0.000400594 | 0.0013412 | 0.0022194 | 0.675 | 0.588 | yes |  |
| patlak | OSIPI Patlak delay=0 | ps | 9 | 0.0001523 | 0.000352332 | 0.000378913 | 0.000164531 | 0.000378913 | 0.000479072 | 0.926 | 0.791 | yes |  |
| patlak | OSIPI Patlak delay=0 | vp | 9 | 0.000612282 | 0.00138231 | 0.00176804 | 0.000539772 | 0.00176798 | 0.00197796 | 1.134 | 0.894 | yes |  |
| patlak | OSIPI Patlak delay=5 | ps | 9 | 0.0210436 | 0.0400774 | 0.0402299 | 0.000164531 | 0.000378913 | 0.000479072 | 127.901 | 83.975 | no | delay fitting not implemented yet; run shown for gap visibility |
| patlak | OSIPI Patlak delay=5 | vp | 9 | 0.0894346 | 0.166622 | 0.169393 | 0.000539772 | 0.00176798 | 0.00197796 | 165.690 | 85.640 | no | delay fitting not implemented yet; run shown for gap visibility |
| 2cxm | OSIPI 2CXM delay=0 | ve | 24 | 0.0762075 | 0.0956351 | 0.8 | 0.00136397 | 0.00553435 | 0.0158681 | 55.872 | 50.416 | no | nonfinite fit failures=3 |
| 2cxm | OSIPI 2CXM delay=0 | vp | 24 | 0.0712617 | 0.164448 | 0.235699 | 0.00132148 | 0.00617296 | 0.0185702 | 53.926 | 12.692 | no | nonfinite fit failures=3 |
| 2cxm | OSIPI 2CXM delay=0 | fp | 24 | 38.7517 | 65.5378 | 123.782 | 0.21838 | 1.0296 | 1.94074 | 177.451 | 63.781 | no | nonfinite fit failures=3 |
| 2cxm | OSIPI 2CXM delay=0 | ps | 24 | 0.0516094 | 0.109517 | 0.110708 | 0.002023 | 0.0140723 | 0.0186095 | 25.511 | 5.949 | no | nonfinite fit failures=3 |
| 2cxm | OSIPI 2CXM delay=5 | ve | 24 | 0.039351 | 0.100049 | 0.108512 | 0.00136397 | 0.00553435 | 0.0158681 | 28.850 | 6.838 | no | delay fitting not implemented yet; run shown for gap visibility |
| 2cxm | OSIPI 2CXM delay=5 | vp | 24 | 0.0834619 | 0.0989807 | 0.7151 | 0.00132148 | 0.00617296 | 0.0185702 | 63.158 | 38.508 | no | delay fitting not implemented yet; run shown for gap visibility |
| 2cxm | OSIPI 2CXM delay=5 | fp | 24 | 86.5947 | 152.959 | 745.664 | 0.21838 | 1.0296 | 1.94074 | 396.533 | 384.217 | no | delay fitting not implemented yet; run shown for gap visibility |
| 2cxm | OSIPI 2CXM delay=5 | ps | 24 | 0.0544201 | 0.11743 | 0.176333 | 0.002023 | 0.0140723 | 0.0186095 | 26.901 | 9.475 | no | delay fitting not implemented yet; run shown for gap visibility |
| 2cum | OSIPI 2CUM delay=0 | vp | 27 | 0.00249536 | 0.00403755 | 0.00420339 | 0.000584325 | 0.0018815 | 0.00340002 | 4.270 | 1.236 | no | nonfinite fit failures=8 |
| 2cum | OSIPI 2CUM delay=0 | fp | 27 | 30.6815 | 39.5714 | 39.5723 | 0.359035 | 1.45653 | 4.49326 | 85.455 | 8.807 | no | nonfinite fit failures=8 |
| 2cum | OSIPI 2CUM delay=0 | ps | 27 | 0.000565754 | 0.00168419 | 0.00177583 | 0.000528323 | 0.00140357 | 0.00173558 | 1.071 | 1.023 | no | nonfinite fit failures=8 |
| 2cum | OSIPI 2CUM delay=5 | vp | 27 | 0.0145501 | 0.032752 | 0.13611 | 0.000584325 | 0.0018815 | 0.00340002 | 24.901 | 40.032 | no | delay fitting not implemented yet; run shown for gap visibility; nonfinite fit failures=9 |
| 2cum | OSIPI 2CUM delay=5 | fp | 27 | 95.3876 | 213.126 | 1195 | 0.359035 | 1.45653 | 4.49326 | 265.677 | 265.954 | no | delay fitting not implemented yet; run shown for gap visibility; nonfinite fit failures=9 |
| 2cum | OSIPI 2CUM delay=5 | ps | 27 | 0.00124911 | 0.00326032 | 0.00488162 | 0.000528323 | 0.00140357 | 0.00173558 | 2.364 | 2.813 | no | delay fitting not implemented yet; run shown for gap visibility; nonfinite fit failures=9 |
| t1_linear | OSIPI T1 (brain+quiba+prostate) | r1 | 171 | 0.0184068 | 0.0558289 | 0.428272 | 0.0184068 | 0.0564628 | 0.428272 | 1.000 | 1.000 | yes |  |
