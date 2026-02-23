function [results, B_vars] = B_AIF_fitting_func(results_a_path, start_time, ...
    end_time,start_injection,end_injection,fit_aif,import_aif_path,time_resolution, ...
    timevectpath, RUNA_struct, save_output)

% B_AIF_fitting_func - This file is used to apply a model fitting to the
% arterial input function. AIF is fit to a bi-exponential model with a
% linear ramp up (from baseline to peak)
%
% Inputs:
%  results_a_path     - *.mat Results from part A
%  start_time         - start time to restrict analysis to (in minutes, 0
%                       if all)
%  end_time           - end time to restrict analysis to (in minutes, 0 if
%                       all)
%  start_injection    - start of contrast injection (in minutes)
%  end_injection      - end of contrast injection (in minutes)
%  fit_aif            - fit AIF to model (bool, 0 for reference region)
%  import_aif_path    - import AIF from .mat file (string)
%  time_resolution    - time resolution of dynamic scan (in minutes)
%  timevectpath       - import manual time vector from .mat file (string)
%
%
% The script loads the data arrays generated from A_make_R1maps_func().
% Then one can
%
% a) fit to a model.
% b) limit the time interval from which further data fitting is performed,
% as well as define a unique time vector.
% c) There is an option import a population or averaged AIF from multiple
% datasets should you desire. This may be useful if you want to get better
% SNR or comparing multiday studies.
%
% Requires:
% B_AIF_fitting_func.m
% AIFbiexpfithelp.m
% DataHash.m
% AIFbiexpcon.m
% parse_preference_file.m
% timeresmaker.m *
% setupxdata2.m *
% setupxdata3.m *
% prepandsave.m *
%
% Thomas Ng
% Caltech
% December 2011, Jun 2014
% Updated Dec 2013 Sam Barnes

%% Toggle options
% If you have a threshold value for noise (testing).
threshold = 0;
if nargin<11
    save_output=true;
end

%% DO NOT ALTER BELOW UNLESS YOU KNOW WHAT YOU ARE DOING

%% 1. Load the data array from previous script

if ~exist('RUNA_struct','var')
    load(results_a_path);
else
    Adata = RUNA_struct;
end

% unload the variables from previous data array
quant    = Adata.quant;
rootname = Adata.rootname;
Cp       = double(Adata.Cp);
Ct       = double(Adata.Ct);
if (start_injection == -1 || end_injection == -1)
    start_injection = Adata.start_injection*time_resolution;
    end_injection = Adata.end_injection*time_resolution;
end

% We also load the Rawdata for raw curve fitting if necessary
Stlv    = double(Adata.Stlv);
Sttum   = double(Adata.Sttum);
Sss     = double(Adata.Sss);
Ssstum  = double(Adata.Ssstum);

results  = '';

% update output path to be same as location of input
[PathName1,~,~] = fileparts(results_a_path);

% Log input results
log_path = fullfile(PathName1, ['B_' rootname 'test_R1info.log']);
if exist(log_path, 'file')==2
    delete(log_path);
end
diary(log_path);
fprintf('************** User Input **************\n\n');
disp('User selected part A results: ');
disp(results_a_path);
Opt.Input = 'file';
% try
%     a_md5 = DataHash(results_a_path, Opt);
% catch
%     disp('Problem using md5 hashing. Will continue');
%     a_md5 = 'error';
% end
% fprintf('File MD5 hash: %s\n\n', a_md5)
disp('User selected start time (min): ');
disp(start_time);
disp('User selected end time (min): ');
disp(end_time);
disp('User selected start injection (min): ');
disp(start_injection);
disp('User selected end injection (min): ');
disp(end_injection);
disp('User selected fit AIF to model: ');
disp(fit_aif);
disp('User selected import AIF path: ');
fprintf('%s\n\n',import_aif_path);

if ~isempty(timevectpath)
    disp('User selected import time vector path: ');
    fprintf('%s\n\n',timevectpath);
else
    disp('User selected time resolution (sec)');
    disp(time_resolution*60);
end

if quant
    disp('User selected quantification: ie. use T1 maps');
else
    disp('User selected no quantification: raw signal data only');
end
fprintf('************** End User Input **************\n\n\n');

disp('Starting Part B - Fitting AIF')
disp(datestr(now))
disp(' ');
tic

%% 2. time management (Change the starter, timeend if needed)
%% b) Time management

if ~isempty(timevectpath)
    external = load(timevectpath);
    if isfield(external,'timer')
        timer = external.timer;
        
        if length(timer) ~= prod(size(timer))
            disp('Time vector not a vector');
            return
        end
    else
        disp('No time vector found in selected file')
        return
    end
    
    % Check that time vector is same size as data. If shorter, extrapolate
    % using the last time interval as guide.

    if length(timer) > size(Cp,1)
        timer = timer(1:size(Cp,1));
    elseif length(timer) < size(Cp,1)
        lasttimeres = timer(end)-timer(end-1);
        
        while length(timer < size(Cp,1))
            timer(end+1) = timer(end)+lasttimeres;
        end
    else
        disp('Imported time vector same size as data');
    end
    
    if max(timer) > 100
        % Most likely in seconds, convert to minutes
        disp('Imported time vector appears to be in seconds, converting to minutes');
        timer = timer./60;
    end
else   
    %Generate time curve
    timer   = 0:time_resolution:time_resolution*(size(Cp,1)-1);

end

%Restrict to specified times
if(start_time)
    timecheck = abs(timer-start_time);
    start_time = find(timecheck == min(timecheck));
else
    start_time = 1;
end

if(end_time)
    timecheck = abs(timer-end_time);
    end_time = find(timecheck == min(timecheck));
else
    end_time = numel(timer);
end


%% Setup data to allow lsqcurvefit to work
xdata{1}.timer = timer(start_time:end_time)';
timer          = xdata{1}.timer;
CpROI          = mean(Cp,2);
CpROI          = CpROI(start_time:end_time)';
xdata{1}.Cp    = CpROI;

StlvROI        = mean(Stlv,2);
StlvROI        = StlvROI(start_time:end_time)';
xdata{1}.Stlv  = StlvROI;

% threshold (Remove noise manually)
if(threshold)
    ind = find(CpROI > threshold);
    
    for j = 1:numel(ind)
        CpROI(ind(j)) = [];
        timer(ind(j)) = [];
        Cp1(ind(j), :)= [];
        Cp2(ind(j),:) = [];
    end
    
    xdata{1}.Cp    = CpROI;
    xdata{1}.timer = timer;
end




%% 4. Define the step function
xdata{1}.step = [start_injection end_injection];

%% 5. Either Fit the curve to the model, keep the curve as raw data, or 
%     replace the curve with an AIF from another dataset.
% NEED TO WORK ON FITTING TO BE MORE ROBUST

M{1} = '';
aif_name = '';
if isempty(import_aif_path)
    if(fit_aif==1)
        xdata{1}.fittingAU = false;
        [Cp_fitted xAIF xdataAIF] = AIFbiexpfithelp(xdata, 1);
        Cp_use = Cp_fitted;
 
        M{2} = 'Fitted Curve';
        aif_name = 'fitted';
        
        %Fit raw data curve
        Cptemp = xdata{1}.Cp;
        xdata{1}.Cp = xdata{1}.Stlv;
        xdata{1}.fittingAU = true;
        [Stlv_fitted, ~, ~] = AIFbiexpfithelp(xdata, 1);
        xdata{1}.Cp = Cptemp;
        Stlv_use = Stlv_fitted;
    elseif(fit_aif==2)
        Cp_use = CpROI;
        M{2} = 'Using Raw Curve';
        aif_name = 'raw';
        
        Stlv_use = StlvROI;
    end
else
    external = load(import_aif_path);
    if isfield(external,'Cp_use')
        Cp_use = external.Cp_use;
        Stlv_use = external.Stlv_use;
        import_timer = timer;
        import_start = external.start_injection;
    elseif isfield(external,'Bdata')
        if isfield(external.Bdata,'Cp_use')
            Cp_use = external.Bdata.Cp_use;
            Stlv_use = external.Bdata.Stlv_use;
            import_timer = external.Bdata.timer;
            import_start = external.Bdata.start_injection;
        elseif isfield(external.Bdata,'xdata')
            Cp_use = external.Bdata.xdata{1}.Cp;
            Stlv_use = external.Bdata.xdata{1}.Stlv;
            import_timer = external.Bdata.xdata{1}.timer;
            import_start = external.Bdata.xdata{1}.step(1);
        end
    elseif isfield(external,'xdata')
        Cp_use = external.xdata{1}.Cp;
        Stlv_use = exernal.xdata{1}.Stlv_use;
        import_timer = external.xdata{1}.timer;
        import_start = external.xdata{1}.step(1);
    elseif endsWith(import_aif_path, 'csv')
        S_ratio = external .* (1 - exp(-Adata.tr / Adata.blood_t1)) / (1 - exp(-Adata.tr/Adata.blood_t1)*cosd(Adata.fa));
        R1 = (1 / Adata.tr) .* log((1 - S_ratio .* cosd(Adata.fa)) ./ (1 - S_ratio));
        Cp_use = (R1 - 1/Adata.blood_t1) / (Adata.relaxivity*(1-Adata.hematocrit));
        Cp_use = Cp_use';
        Stlv_use = external';
        import_timer = timer;
        [~, max_index] = max(Stlv_use);
        import_start = timer(max_index);
    else
        disp('No Cp curve found in selected file')
        return
    end
    % Check length, if different try applying time constraints
    if numel(timer)<numel(Cp_use)
        Cp_use = Cp_use(start_time:end_time);
        Stlv_use = Stlv_use(start_time:end_time);
        if numel(timer)~=numel(Cp_use)
            disp('Imported AIF has different length than data, try using time limits');
            return;
        end
    end
    % Check if needs to be time shifted a bit, allowed to shift 10 time
    % points
    aif_start_index = find(abs(import_timer - import_start) == min(abs(import_timer - import_start)));
    data_start_time = xdata{1}.step(1);
    data_start_index = find(abs(timer - data_start_time) == min(abs(timer - data_start_time)));
    if aif_start_index~=data_start_index
        import_shift = data_start_index-aif_start_index;
        Cp_use = circshift(Cp_use,[1 import_shift]);
        if import_shift<0
            % Pad end with repeated value
            Cp_use(end+import_shift:end) = Cp_use(end-1+import_shift);
        else
            % Pad beginning with zeros
            Cp_use(1:import_shift) = 0;
        end
        disp(['Imported AIF has been shifted by ' num2str(import_shift) ' images to match data']);
    else
        disp(['Imported AIF not shifted']);
    end
    
    M{2} = 'Imported Curve';
    aif_name = 'imported';
end

% 5.5 Plot the results
b = figure;
subplot(1,2,1)
plot(timer,CpROI,'r.');
hold on;
plot(timer, Cp_use,'b');
disp('AIF mmol:')
perLine = 14;
fmt = [repmat('%8.4f ', 1, perLine), '\n'];
fprintf(fmt, CpROI);
if mod( length(CpROI), perLine) ~= 0; fprintf('\n'); end

M{1} = 'Original Plasma Curve';
% M{2} = 'Selected Curve';

legend(M);
root_modified = rootname;
root_modified(end)='';
hold off;
title([root_modified ' - AIF Bi-Exponential, Linear Upslope'], 'Interpreter', 'none');
ylabel('Concentration (mM)');
xlabel('Time (min)');

subplot(1,2,2)
plot(timer,StlvROI,'r.');
hold on;
plot(timer, Stlv_use,'b');

M{1} = 'Original Plasma Curve: Raw data';
legend(M);
hold off;
title([root_modified ' - AIF Bi-Exponential, Linear Upslope'], 'Interpreter', 'none');
ylabel('Signal (a.u)');
xlabel('Time (min)');
saveas(b, fullfile(PathName1, [rootname 'AIF_fitting.fig']));

%% 6.  Save for voxel by voxel fitting, setup ready to go
xdata{1}.timer = timer;
xdata{1}.Ct    = Ct(start_time:end_time,:);
numvoxels      = size(Ct,2);
xdata{1}.Cp = Cp_use;

xdata{1}.Sttum = Sttum(start_time:end_time,:);
xdata{1}.Stlv  = Stlv_use;

%% 7. Setup  Bresults to output
Bdata.CpROI         = CpROI;
Bdata.Cp_use        = Cp_use;
Bdata.aif_name      = aif_name;
Bdata.end_injection = end_injection;
Bdata.end_time      = end_time;
Bdata.log_path      = log_path;
Bdata.numvoxels     = numvoxels;
Bdata.results_a_path= results_a_path; % Location of associated A data
Bdata.start_injection=start_injection;
Bdata.start_time    = start_time;
Bdata.threshold     = threshold;
Bdata.time_resolution=time_resolution;
Bdata.timer         = timer;
Bdata.xdata         = xdata;

% Results from A that need to be passed through
Bdata.rootname    = Adata.rootname;
Bdata.R1tTOI      = Adata.R1tTOI(start_time:end_time,:);
Bdata.R1tLV       = Adata.R1tLV(start_time:end_time,:);
Bdata.deltaR1LV   = Adata.deltaR1LV(start_time:end_time,:);
Bdata.deltaR1TOI  = Adata.deltaR1TOI(start_time:end_time,:);
Bdata.T1TUM       = Adata.T1TUM;
Bdata.tumind      = Adata.tumind;
Bdata.dynam_name  = Adata.dynam_name;
Bdata.currentCT   = Adata.currentCT;
Bdata.res         = Adata.res;
Bdata.relaxivity  = Adata.relaxivity;
Bdata.hdr         = Adata.hdr;
Bdata.quant       = quant;
% for AUC
Bdata.Sss         = Adata.Sss;
Bdata.Ssstum      = Adata.Ssstum;



results = fullfile(PathName1, ['B_' rootname aif_name '_R1info.mat']);

if save_output==true
    save(results, 'Bdata','-v7.3');

    Opt.Input = 'file';
% try
%     mat_md5 = DataHash(results, Opt);
% catch
%     disp('Problem using md5 hashing. Will continue');
%     mat_md5 = 'error';
% end
    disp(' ')
    disp('MAT results saved to: ')
    disp(results)
% disp(['File MD5 hash: ' mat_md5])
else
    B_vars = Bdata;
end

disp(' ');
disp('Finished B');
disp(datestr(now))
toc
diary off;

% Keep log consistent naming with data file
movefile(fullfile(PathName1, ['B_' rootname 'test_R1info.log']) , fullfile(PathName1, ['B_' rootname aif_name '_R1info.log']));
