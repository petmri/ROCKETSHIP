function prefs = default_dce_fit_prefs()
% default_dce_fit_prefs Deterministic fitting preferences for unit tests.

prefs.lower_limit_ktrans = 1e-7;
prefs.upper_limit_ktrans = 2;
prefs.initial_value_ktrans = 2e-4;

prefs.lower_limit_ve = 0.02;
prefs.upper_limit_ve = 1;
prefs.initial_value_ve = 0.2;

prefs.lower_limit_vp = 1e-3;
prefs.upper_limit_vp = 1;
prefs.initial_value_vp = 0.02;

prefs.lower_limit_fp = 1e-3;
prefs.upper_limit_fp = 100;
prefs.initial_value_fp = 0.2;

prefs.lower_limit_tp = 0;
prefs.upper_limit_tp = 1e6;
prefs.initial_value_tp = 0.05;

prefs.lower_limit_tau = 0;
prefs.upper_limit_tau = 100;
prefs.initial_value_tau = 0.01;

prefs.lower_limit_ktrans_RR = 1e-7;
prefs.upper_limit_ktrans_RR = 2;
prefs.initial_value_ktrans_RR = 0.1;

prefs.TolFun = 1e-12;
prefs.TolX = 1e-8;
prefs.MaxIter = 400;
prefs.MaxFunEvals = 2000;
prefs.Robust = 'off';
end
