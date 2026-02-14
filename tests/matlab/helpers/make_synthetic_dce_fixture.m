function fixture = make_synthetic_dce_fixture()
% make_synthetic_dce_fixture Synthetic DCE inputs used for tests/baselines.

timer = (0:0.1:5)';
Cp = (timer.^2) .* exp(-timer / 0.9);
Cp = Cp ./ max(Cp);
Cp = Cp + 0.03;

fixture.timer = timer;
fixture.Cp = Cp;

fixture.ktrans = 0.03;
fixture.ve = 0.25;
fixture.vp = 0.04;
fixture.fp = 0.15;
fixture.tau = 0.08;

fixture.R1o = 1.30;
fixture.R1i = 0.65;
fixture.r1 = 3.40;
fixture.fw = 0.80;

fixture.Ct_tofts = model_tofts_cfit(fixture.ktrans, fixture.ve, Cp, timer);
fixture.Ct_extended_tofts = model_extended_tofts_cfit(fixture.ktrans, fixture.ve, fixture.vp, Cp, timer);
fixture.Ct_patlak = model_patlak_cfit(fixture.ktrans, fixture.vp, Cp, timer);
fixture.Ct_2cxm = model_2cxm_cfit(fixture.ktrans, fixture.ve, fixture.vp, fixture.fp, Cp, timer);
fixture.R1t_fxr = model_fxr_cfit(fixture.ktrans, fixture.ve, fixture.tau, Cp, timer, ...
    fixture.R1o, fixture.R1i, fixture.r1, fixture.fw);
end
