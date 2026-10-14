function version_str = report_gpufit_version()
% REPORT_GPUFIT_VERSION - Installed Gpufit library version, for run-log audit trails.
%
% Mirrors python/accel_backend.py's pygpufit_version reporting, so a log from either
% implementation shows which Gpufit build actually ran the fit.
try
    version_str = char(gpufit_version());
catch
    version_str = 'unknown';
end
