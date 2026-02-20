function baseline = export_parity_baseline(outputDir)
% export_parity_baseline Export canonical MATLAB outputs for parity checks.

repoRoot = add_rocketship_paths();

if nargin < 1 || isempty(outputDir)
    outputDir = fullfile(repoRoot, 'tests', 'contracts', 'baselines');
end

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

fixture = make_synthetic_dce_fixture();
prefs = default_dce_fit_prefs();

baseline = struct();
baseline.meta = struct();
baseline.meta.version = 'v1';
baseline.meta.generated_utc = datestr(now, 30);

baseline.dce.forward = struct();
baseline.dce.forward.timer = fixture.timer;
baseline.dce.forward.Cp = fixture.Cp;
baseline.dce.forward.tofts = fixture.Ct_tofts;
baseline.dce.forward.extended_tofts = fixture.Ct_extended_tofts;
baseline.dce.forward.patlak = fixture.Ct_patlak;
baseline.dce.forward.vp = fixture.Ct_vp;
baseline.dce.forward.tissue_uptake = fixture.Ct_tissue_uptake;
baseline.dce.forward.twocxm = fixture.Ct_2cxm;
baseline.dce.forward.fxr = fixture.R1t_fxr;

baseline.dce.params = struct();
baseline.dce.params.ktrans = fixture.ktrans;
baseline.dce.params.ve = fixture.ve;
baseline.dce.params.vp = fixture.vp;
baseline.dce.params.fp = fixture.fp;
baseline.dce.params.tp = fixture.tp;
baseline.dce.params.tau = fixture.tau;
baseline.dce.params.R1o = fixture.R1o;
baseline.dce.params.R1i = fixture.R1i;
baseline.dce.params.r1 = fixture.r1;
baseline.dce.params.fw = fixture.fw;

baseline.dce.inverse = struct();
baseline.dce.inverse.patlak_linear = model_patlak_linear(fixture.Ct_patlak, fixture.Cp', fixture.timer);
baseline.dce.inverse.tofts_fit = model_tofts(fixture.Ct_tofts, fixture.Cp', fixture.timer, prefs);
baseline.dce.inverse.extended_tofts_fit = model_extended_tofts(fixture.Ct_extended_tofts, fixture.Cp', fixture.timer, prefs);
baseline.dce.inverse.vp_fit = model_vp(fixture.Ct_vp, fixture.Cp', fixture.timer, prefs);
baseline.dce.inverse.tissue_uptake_fit = model_tissue_uptake(fixture.Ct_tissue_uptake, fixture.Cp', fixture.timer, prefs);
baseline.dce.inverse.twocxm_fit = model_2cxm(fixture.Ct_2cxm, fixture.Cp', fixture.timer, prefs);
baseline.dce.inverse.fxr_fit = model_fxr(fixture.R1t_fxr, fixture.Cp', fixture.timer, ...
    fixture.R1o, fixture.R1i, fixture.r1, fixture.fw, prefs);

meanAIF = linspace(0, 1.1, 14)';
bolusTime = 3;
timeVect = (0:0.1:1.8)';
concentrationArray = reshape(linspace(0.05, 0.6, 2 * 2 * numel(timeVect)), [2, 2, numel(timeVect)]);

[meanAIFAdjusted, timeVectAdj, concentrationAdj, meanSignal] = import_AIF(meanAIF, bolusTime, timeVect, concentrationArray, 3.4, 0.03);
[meanAIFPrev, timeVectPrev, concentrationPrev] = previous_AIF(meanAIFAdjusted, meanSignal, bolusTime, timeVectAdj, concentrationAdj);

baseline.dsc = struct();
baseline.dsc.import_aif = struct('meanAIF_adjusted', meanAIFAdjusted, 'time_vect', timeVectAdj, ...
    'concentration_array', concentrationAdj, 'meanSignal', meanSignal);
baseline.dsc.previous_aif = struct('meanAIF_adjusted', meanAIFPrev, 'time_vect', timeVectPrev, ...
    'concentration_array', concentrationPrev);

ssvdConc = zeros(2, 2, 10);
timeIndex = 0:9;
for ix = 1:2
    for iy = 1:2
        ssvdConc(ix, iy, :) = exp(-((timeIndex - (2 + ix + iy / 2)).^2) / 6);
    end
end
ssvdAIF = exp(-((timeIndex - 2).^2) / 4)';
ssvdOutDir = tempname;
mkdir(ssvdOutDir);
[ssvdCBF, ssvdCBV, ssvdMTT] = DSC_convolution_sSVD(ssvdConc, ssvdAIF, 0.1, 0.73, 1.04, 20, 1, [ssvdOutDir filesep]);
baseline.dsc.ssvd_deconvolution = struct('CBF', ssvdCBF, 'CBV', ssvdCBV, 'MTT', ssvdMTT);

te = [10; 20; 40; 60];
trueT2 = 85;
rho = 900;
siT2 = rho .* exp(-te ./ trueT2);
baseline.parametric = struct();
baseline.parametric.t2_linear_fast = fitParameter(te, 't2_linear_fast', siT2, 0, '', 0, '', 0, 0);

fa = [2; 5; 10; 15];
tr = 8;
trueT1 = 1300;
m0 = 1100;
theta = fa .* (pi / 180);
siT1 = m0 .* ((1 - exp(-tr ./ trueT1)) .* sin(theta)) ./ (1 - exp(-tr ./ trueT1) .* cos(theta));
baseline.parametric.t1_fa_linear_fit = fitParameter(fa, 't1_fa_linear_fit', siT1, tr, '', 0, '', 0, 0);
baseline.parametric.t1_fa_fit = fitParameter(fa, 't1_fa_fit', siT1, tr, '', 0, '', 0, 0);

matPath = fullfile(outputDir, 'matlab_reference_v1.mat');
jsonPath = fullfile(outputDir, 'matlab_reference_v1.json');

save(matPath, 'baseline', '-v7');

jsonText = jsonencode(baseline);
fid = fopen(jsonPath, 'w');
if fid == -1
    error('Unable to open baseline json file for writing: %s', jsonPath);
end
fwrite(fid, jsonText, 'char');
fclose(fid);

fprintf('Wrote baseline outputs:\n%s\n%s\n', matPath, jsonPath);
end
