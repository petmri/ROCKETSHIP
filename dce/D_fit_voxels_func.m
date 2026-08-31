function results = D_fit_voxels_func(results_b_path, dce_model, ...
    time_smoothing,time_smoothing_window,xy_smooth_size,number_cpus, ...
    roi_list,fit_voxels,neuroecon, outputft, B_vars, save_output)

% D_fit_voxels_func - Fit DCE curve to various models on a voxel by voxel
% or ROI basis
%
% Inputs:
%  results_b_path     - *.mat Results from part B
%  dce_model          - Select fitting model
%                       'ex_tofts' = tofts with vascular compartment
%                       'tofts' = tofts without vascular compartment
%                       'fxr' = "shutter speed"
%                       'sauc' = not implemented
%                       'ss' = not implemented
%                       'fractal' = not implemented
%                       'auc' = area under the curve
%                       'auc_rr' = not implemented
%                       'nested' = series of nested model
%                       'patlak' = two parameter model with no backflux
%                       'tissue_uptake' = three parameter assumes cp>>ct
%                       '2cxm' = two compartment exchange model
%                       'fxl_RR' = Fast exchange limit tofts with Reference
%                       region'
%  time_smoothing     - type of time smoothing
%                       'none' = no smoothing
%                       'moving' = moving average
%                       'rlowess' = robust local regression
%  time_smoothing_window - size of time smoothing window (time points)
%  xy_smooth_size     -	sigma of the Gaussian low pass smooth function
%  number_cpus        - number of cpu cores for parallel processing
%  roi_list           - paths to ROIs that specify homogenous regions to
%                       calculate a single DCE fit for. Values >0
%                       considered in the ROI, values <=0 considered
%                       outside the ROI
%  fit_voxels         - perform DCE fit on individual voxels
%  neuroecon          - perform processing on neuroecon server
%  batch              - prep for batch procesing only
%  outputft           - output filetype
%
% The script loads the data arrays generated from B_AIF_fitting_func().
% Then it will fit a DCE curve according to various models
%
% Requires:
% D_fit_voxels_func.m
% DataHash.m
% FXLfit_generic.m
% ProgressBar.m
% FXLStep1AIFhelper.m
% FXLStep1AIFhelper_vp.m
% fxr_helper.m
% parse_preference_file.m
% niftitools
%
% Samuel Barnes
% Caltech
% December 2013
% Updated Thomas Ng Jan 2014


% Toggle options
%************************
r2filter = 0;		% Filter out all fits with r2 < r2filter
if nargin < 12
    save_output=true;
end
% close_pool = 0;		% Close matlabpool when done with processing
% End options
%************************

% a) Load the data files
if ~exist('B_vars', 'var') || isempty(B_vars)
    if ~exist(results_b_path, 'file')
        warning( 'File does not exist' );
        disp(results_b_path);
        return;
    end
    load(results_b_path);
else
    disp('Using B_vars from input');
    Bdata = B_vars;
end

quant = Bdata.quant;

% Setup DCE model string
dce_model_string = {};
dce_model_list = {};

if quant
    if dce_model.tofts
        dce_model_string{end+1} = 'Tofts';
        dce_model_list{end+1} = 'tofts';
    end
    if dce_model.ex_tofts
        dce_model_string{end+1} = 'Tofts w/ Vp';
        dce_model_list{end+1} = 'ex_tofts';
    end
    if dce_model.fxr
        dce_model_string{end+1} = 'FXR';
        dce_model_list{end+1} = 'fxr';
    end
    if dce_model.nested
        dce_model_string{end+1} = 'Nested Models';
        dce_model_list{end+1} = 'nested';
    end
    if dce_model.patlak
        dce_model_string{end+1} = 'Patlak';
        dce_model_list{end+1} = 'patlak';
    end
    if dce_model.tissue_uptake
        dce_model_string{end+1} = 'Tissue Uptake';
        dce_model_list{end+1} = 'tissue_uptake';
    end
    if dce_model.two_cxm
        dce_model_string{end+1} = 'Two Compartment Exchange';
        dce_model_list{end+1} = '2cxm';
    end
    if dce_model.FXL_rr
        dce_model_string{end+1} = 'FXL Reference Region';
        dce_model_list{end+1} = 'FXL_rr';
    end
end
if dce_model.fractal
    dce_model_string{end+1} = 'Fractal Metric';
    dce_model_list{end+1} = 'fractal';
end
if dce_model.auc
    dce_model_string{end+1} = 'Area under curve';
    dce_model_list{end+1} = 'auc';
end

results = {};
for model_index=1:numel(dce_model_list)
    cur_dce_model = dce_model_list{model_index};
    
    % Load the variables as needed.
    rootname    = Bdata.rootname;
    R1tTOI      = Bdata.R1tTOI;
    R1tLV       = Bdata.R1tLV;
    deltaR1LV   = Bdata.deltaR1LV;
    deltaR1TOI  = Bdata.deltaR1TOI;
    
    T1TUM       = Bdata.T1TUM;
    tumind      = Bdata.tumind;
    dynam_name  = Bdata.dynam_name;
    currentimg  = Bdata.currentCT;
    res         = Bdata.res;
    relaxivity  = Bdata.relaxivity;
    hdr         = Bdata.hdr;
    %     sliceloc    = Adata.sliceloc;
    xdata       = Bdata.xdata;
    start_time  = Bdata.start_time;
    end_time    = Bdata.end_time;
    numvoxels   = Bdata.numvoxels;
    
    if start_time == 0
        start_time = 1;
    end
    if end_time == 0
        end_time = size(R1tTOI,1);
    end
    
    % Load and prep raw signal files if required
    if strcmp(cur_dce_model,'auc')
        xdata{1}.start_injection    = Bdata.start_injection;
        xdata{1}.end_injection      = Bdata.end_injection;
        xdata{1}.Ssstum             = Bdata.Ssstum;
        xdata{1}.Sss                = Bdata.Sss;
    end
    
    % update output path to be same as location of input
    [PathName,~,~] = fileparts(results_b_path);
    
    % Log input results
    results_base = fullfile(PathName, [rootname '_' cur_dce_model '_fit']);
    log_path = [results_base,'.log'];
    if exist(log_path, 'file')==2
        delete(log_path);
    end
    diary(log_path);
    fprintf('************** User Input **************\n\n');
    disp('User selected part B results: ');
    disp(results_b_path);
    Opt.Input = 'file';
    % try
    %     b_md5 = DataHash(results_b_path, Opt);
    % catch
    %     disp('Problem using md5 hashing. Will continue');
    %     b_md5 = 'error';
    % end
    % fprintf('File MD5 hash: %s\n\n', b_md5)
    disp('User selected dce model: ');
    fprintf('%s\n\n',dce_model_string{model_index});
    disp('User selected time smoothing model: ');
    fprintf('%s\n\n',time_smoothing);
    disp('User selected time smoothing window size: ');
    disp(time_smoothing_window);
    disp('User selected XY smooth size (sigma)');
    disp(xy_smooth_size);
    
    % Auto detect GPU
    try
        USE_GPU = GpufitCudaAvailableMex;
    catch
        USE_GPU = 0;
    end

    if USE_GPU
        disp("Gpufit detected. GPU will be utilized for voxel fitting.")
    else
        disp("Gpufit not detected. Defaulting to CPU.")
    end
    
    % Setup matlabpool size if using CPU
    gpu_prefs = parse_preference_file('dce_preferences.txt',0,{'force_cpu'},{0});
    FORCE_CPU = str2num(gpu_prefs.force_cpu);
    if FORCE_CPU
        USE_GPU = 0;
    end
    
    if ~USE_GPU
        disp('User selected number of CPU cores');
        c = parcluster('local'); % build the 'local' cluster object
        n = c.NumWorkers;        % get the number of workers
        if(number_cpus <= n) && number_cpus > 0
        elseif(number_cpus == 0)
            number_cpus = n;
        else
            number_cpus = max(n+number_cpus, 1);
        end
        disp(number_cpus);
    end
    
    disp('User selected ROI list');
    [nrows,ncols]= size(roi_list);
    for row=1:nrows
        disp(roi_list{row,:})
    end
    disp(' ');
    disp('User selected fit individual voxels');
    disp(fit_voxels);
    disp('User selected use neuroecon');
    disp(neuroecon);
    fprintf('************** End User Input **************\n\n\n');
    
    disp('Starting Part D - Fitting Voxels/ROIs')
    disp(datestr(now))
    disp(' ');
    %%%%%%%%%%%%%%%%tic
    
    % Start processing
    xdata{1}.numvoxels = numvoxels;
    %disp(['Fitting data using the ' 'dce' ' model']);
    
    % Open pool if not open or improperly sized
    try
        if ~USE_GPU && ~isempty(ver('parallel'))
            r_prefs = parse_preference_file('dce_preferences.txt',0,{'use_matlabpool'},{0});
            if str2num(r_prefs.use_matlabpool)
                if matlabpool('size')~= number_cpus
                    % Do not launch pool with diary on, locks the log file
                    diary off;
                    if matlabpool('size')>0
                        matlabpool close;
                    end
                    matlabpool('local', number_cpus);
                    diary on;
                end
            else
                s = gcp('nocreate');
                % Do not launch pool with diary on, locks the log file
                diary off;
                if isempty(s)
                    parpool('local',number_cpus);
                else
                    if s.NumWorkers~=number_cpus
                        delete(gcp('nocreate'))
                        parpool('local',number_cpus);
                    end
                end
                diary on;
            end


            % Turn off warnings
            pctRunOnAll warning 'off'
        else
            warning('CPU parallel processing disabled. Make sure the MATLAB Parallel Computing Toolkit is installed.')
        end
    catch
        warning('CPU parallel processing disabled. May be a toolkit licensing issue.')
    end
    
    % Substitute R1 data for concentration data in curve to fit
    % FXR model fits to the R1 data directly, not concentrations
    %     if strcmp(cur_dce_model,'fxr')
    %         xdata{1}.Ct = R1tTOI(start_time:end_time,:);
    %     end
    
    % Save original data
    xdata{1}.Ct_original = xdata{1}.Ct;
    
    % Perpare any ROIs
    number_rois = 0;
    if ~isempty(roi_list) && ~strcmp('No Files',cell2mat(roi_list(1)))
        %Sanitize list
        for m=size(roi_list,1):-1:1
            testfile=cell2mat(roi_list(m));
            if ~exist(testfile, 'file')
                % File does not exist.
                warning( 'File does not exist' );
                disp(testfile);
                return;
            end
            for n=(m-1):-1:1
                comparefile=roi_list(n);
                if strcmp(testfile,comparefile)
                    disp( 'Removing duplicates' );
                    disp(comparefile);
                    roi_list(m)=[];
                end
            end
        end
        number_rois = size(roi_list,1);
        roi_name = [];
        roi_ext = [];
        %After sanitizing make sure we have some left
        if number_rois>0
            [~, roi_name, roi_ext] = arrayfun(@(x) fileparts(x{:}), roi_list, 'UniformOutput', false);
            
            %Load ROI, find the selected voxels
            for r=number_rois:-1:1
                single_file=cell2mat(roi_list(r));
                
                if endsWith(single_file, {'.nii', '.nii.gz', '.hdr', '.img'})
                    single_roi = load_untouch_nii(single_file);
                    single_roi = double(single_roi.img);
                    roi_index{r}= find(single_roi > 0);
                elseif endsWith(single_file, '.roi')
                    single_roi = ReadImageJROI(single_file);
                    if strcmp(single_roi.strType,'Polygon') || strcmp(single_roi.strType,'Freehand')
                        roi_image = poly2mask(...
                            single_roi.mnCoordinates(:,2)+0.5,...
                            single_roi.mnCoordinates(:,1)+0.5,...
                            size(currentimg,1),size(currentimg,2));
                        roi_index{r}= find(roi_image > 0);
                        
                    elseif strcmp(single_roi.strType,'Rectangle')
                        roi_image = poly2mask(...
                            [single_roi.vnRectBounds(1:2:3) fliplr(single_roi.vnRectBounds(1:2:3))]+0.5,...
                            [single_roi.vnRectBounds(2) single_roi.vnRectBounds(2) single_roi.vnRectBounds(4) single_roi.vnRectBounds(4)]+0.5,...
                            size(currentimg,1),size(currentimg,2));
                        roi_index{r}= find(roi_image > 0);
                    elseif strcmp(single_roi.strType,'Oval')
                        center_x = (single_roi.vnRectBounds(3)+single_roi.vnRectBounds(1))/2+0.5;
                        radius_x = (single_roi.vnRectBounds(3)-single_roi.vnRectBounds(1))/2;
                        center_y = (single_roi.vnRectBounds(4)+single_roi.vnRectBounds(2))/2+0.5;
                        radius_y = (single_roi.vnRectBounds(4)-single_roi.vnRectBounds(2))/2;
                        
                        alpha = linspace(0, 360, (radius_x+radius_y)*100)';
                        sinalpha = sind(alpha);
                        cosalpha = cosd(alpha);
                        
                        X = center_x + (radius_x * cosalpha);
                        Y = center_y + (radius_y * sinalpha);
                        
                        roi_image = poly2mask(X,Y,size(currentimg,1),size(currentimg,2));
                        
                        roi_index{r}= find(roi_image > 0);
                    else
                        warning( 'ROI type not supported' );
                        disp(single_roi.strType);
                        return;
                    end
                    % If slice number is present, shift 2D position to
                    % proper z position
                    if isfield(single_roi,'nPosition')
                        if single_roi.nPosition==0
                            single_roi.nPosition = 1;
                        end
                        z_pos = single_roi.nPosition-1;
                        roi_index{r} = roi_index{r} + z_pos*size(currentimg,1)*size(currentimg,2);
                    else
                        error( 'Could not find Z position in ROI file, use latest ImageJ verion' );
                    end
                else
                    warning( 'File type for ROI not supported' );
                    disp(roi_ext(r));
                    return;
                end
            end
            
            original_t1 = zeros(size(currentimg));
            original_ss_signal = zeros(size(currentimg));
            roi_r1 = zeros(number_rois,1);
            roi_ss = zeros(number_rois,1);
            original_timepoint = zeros(size(currentimg));
            roi_series = zeros(size(xdata{1}.Ct,1),number_rois);
            for t=1:size(xdata{1}.Ct,1)
                if strcmp(cur_dce_model,'fxr')
                    original_timepoint(tumind) = R1tTOI(t,:);
                else
                    original_timepoint(tumind) = xdata{1}.Ct(t,:);
                    
                    if strcmp(cur_dce_model,'auc')
                        original_timepoint_signal(tumind) = xdata{1}.Sttum(t,:);
                    end
                end
                % 		original_timepoint(roi_index{r}) = 1e-4;
                % 		imshow(original_timepoint.*20000);
                %Average ROI voxels, insert into time series
                for r=number_rois:-1:1
                    roi_series(t,r) = mean(original_timepoint(roi_index{r}));
                    
                    if strcmp(cur_dce_model,'auc')
                        roi_series_signal(t,r) = mean(original_timepoint_signal(roi_index{r}));
                    end
                end
            end
            % For FXR and AUC
            original_t1(tumind) = T1TUM;
            if strcmp(cur_dce_model,'auc')
                original_ss_signal(tumind) = xdata{1}.Ssstum;
            end
            for r=number_rois:-1:1
                roi_r1(r) = 1./mean(original_t1(roi_index{r}));
                if strcmp(cur_dce_model,'auc')
                    roi_ss(r) = mean(original_ss_signal(roi_index{r}));
                end
            end
            %make backup
            roi_series_original = roi_series;
        end
    end
    
    if ~fit_voxels && number_rois==0
        disp('nothing to fit, select an ROI file or check "fit voxels"');
        return;
    end
    
    % for r=number_rois:-1:1
    % 	figure;
    % 	plot(xdata{1}.timer,roi_series(:,r));
    % end
    % return;
    
    % a1) smoothing in image domain
    if xy_smooth_size>0 && fit_voxels
        % original_timepoint = NaN(size(currentimg));
        original_timepoint = zeros(size(currentimg));
        % make size 3*sigma rounded to nearest odd
        xy_smooth_size_odd = 2.*round((xy_smooth_size*3+1)/2)-1;
        if xy_smooth_size_odd<3
            xy_smooth_size_odd = 3;
        end
        h = fspecial('gaussian', [xy_smooth_size_odd xy_smooth_size_odd],xy_smooth_size);
        for i=1:size(xdata{1}.Ct,1)
            original_timepoint(tumind) = xdata{1}.Ct(i,:);
            smooth_timepoint = zeros(size(original_timepoint));
            for z=1:size(original_timepoint,3)
                smooth_timepoint(:,:,z) = filter2(h, original_timepoint(:,:,z));
            end
            xdata{1}.Ct(i,:) = smooth_timepoint(tumind)';
            
            if strcmp(cur_dce_model,'auc')
                original_timepoint_signal(tumind) = xdata{1}.Sttum(i,:);
                smooth_timepoint_signal = filter2(h, original_timepoint_signal);
                xdata{1}.Sttum(i,:) = smooth_timepoint_signal(tumind);
            end
            
            
        end
    end
    
    % a2) smoothing in time domain
    if ~strcmp(time_smoothing,'none')
        disp('Smoothing time domain');
        
        if fit_voxels
            Ct_all = xdata{1}.Ct;
            p = ProgressBar(size(Ct_all,2));
            parfor i=1:size(Ct_all,2)
                Ct_smooth = Ct_all(:,i);
                if strcmp(time_smoothing,'moving')
                    Ct_smooth = smooth(Ct_smooth,time_smoothing_window,'moving');
                elseif strcmp(time_smoothing,'rlowess')
                    Ct_smooth = smooth(Ct_smooth,time_smoothing_window/size(Ct_smooth,1),'rlowess');
                else
                    % no smoothing
                end
                
                Ct_all(:,i) = Ct_smooth;
                p.progress;
            end
            
            xdata{1}.Ct = Ct_all;
            p.stop;
            
            if strcmp(cur_dce_model,'auc')
                disp('Smoothing time domain for raw signals');
                Sttum_all = xdata{1}.Sttum;
                p = ProgressBar(size(Ct_all,2));
                parfor i=1:size(Ct_all,2)
                    Ct_smooth = Sttum_all(:,i);
                    if strcmp(time_smoothing,'moving')
                        Ct_smooth = smooth(Ct_smooth,time_smoothing_window,'moving');
                    elseif strcmp(time_smoothing,'rlowess')
                        Ct_smooth = smooth(Ct_smooth,time_smoothing_window/size(Ct_smooth,1),'rlowess');
                    else
                        % no smoothing
                    end
                    
                    Sttum_all(:,i) = Ct_smooth;
                    p.progress;
                end
                
                xdata{1}.Sttum = Sttum_all;
                p.stop;
                
            end
            
        end
        
        for r=1:number_rois
            roi_series(:,r);
            roi_smooth = roi_series(:,r);
            if strcmp(time_smoothing,'moving')
                roi_smooth = smooth(roi_smooth,time_smoothing_window,'moving');
            elseif strcmp(time_smoothing,'rlowess')
                roi_smooth = smooth(roi_smooth,time_smoothing_window/size(roi_smooth,1),'rlowess');
            else
                % no smoothing
            end
            
            roi_series(:,r) = roi_smooth;
            
            if strcmp(cur_dce_model,'auc')
                roi_smooth = roi_series_signal(:,r);
                if strcmp(time_smoothing,'moving')
                    roi_smooth = smooth(roi_smooth,time_smoothing_window,'moving');
                elseif strcmp(time_smoothing,'rlowess')
                    roi_smooth = smooth(roi_smooth,time_smoothing_window/size(roi_smooth,1),'rlowess');
                else
                    % no smoothing
                end
                
                roi_series_signal(:,r) = roi_smooth;
            end
        end
    end
    
    % b) curve fitting
    %************************
    disp('  ');
    disp(['Begin making maps for ' cur_dce_model '...']);
    tic
    %%%%%%%%%%%%%%%%%%%%%%%%%
    if(neuroecon)
        if strcmp(cur_dce_model, 'fxr')
            xdata{1}.R1o = 1./T1TUM;
            xdata{1}.R1i = 1./T1TUM;
            xdata{1}.relaxivity = relaxivity;
        end
        
        fitting_results = run_neuroecon_job(number_cpus, xdata, numvoxels, cur_dce_model);
        % x(STARTEND(k,1):STARTEND(k,2),:) = cell2mat(results);
    else
        if number_rois~=0
            disp(['Starting fitting for ' num2str(number_rois) ' ROIs...']);
            % Add subject/session ID to roi_data
            if isfield(Bdata, 'dynam_name')
                % Extract all characters before the 2nd underscore in dynam_name
                underscores = strfind(Bdata.dynam_name, '_');
                if numel(underscores) >= 2
                    roi_data{1}.ID = Bdata.dynam_name(1:underscores(2)-1);
                else
                    roi_data{1}.ID = Bdata.dynam_name;
                end
            elseif isfield(Bdata, 'rootname')
                roi_data{1}.ID = Bdata.rootname;
            else
                roi_data{1}.ID = '';
            end

            roi_data{1}.Cp = xdata{1}.Cp;
            roi_data{1}.timer = xdata{1}.timer;
            roi_data{1}.Ct = roi_series;
            
            if strcmp(cur_dce_model, 'fxr')
                roi_data{1}.R1o = roi_r1;
                roi_data{1}.R1i = roi_r1;
                roi_data{1}.relaxivity = relaxivity;
            end
            
            if strcmp(cur_dce_model, 'auc')
                roi_data{1}.Sttum   = roi_series_signal;
                roi_data{1}.Ssstum  = roi_ss';
                roi_data{1}.Sss     = xdata{1}.Sss;
                roi_data{1}.Stlv    = xdata{1}.Stlv;
                roi_data{1}.start_injection = Bdata.start_injection;
                roi_data{1}.end_injection   = Bdata.end_injection;
            end
            
            [roi_results, roi_residuals] = FXLfit_generic(roi_data, number_rois, cur_dce_model);
            
            disp('ROI fitting done')
            disp(' ')
        end
        if fit_voxels
            disp(['Starting fitting for ' num2str(numvoxels) ' voxels...']);
            
            if strcmp(cur_dce_model, 'fxr')
                xdata{1}.R1o = 1./T1TUM;
                xdata{1}.R1i = 1./T1TUM;
                xdata{1}.relaxivity = relaxivity;
            end

            [fitting_results, voxel_residuals] = FXLfit_generic(xdata, numvoxels, cur_dce_model);
            
            disp('Voxel fitting done')
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%
    toc
    % processing_time = toc;
    % disp(['processing completed in ' datestr(processing_time/86400, 'HH:MM:SS') ' (hr:min:sec)']);
%     if close_pool
%         matlabpool close;
%     end
    
    % c) Save file
    %************************
    fit_data.fit_voxels = fit_voxels;
    fit_data.tumind = tumind;
    fit_data.dynam_name = dynam_name;
    fit_data.PathName = PathName;
    fit_data.time_smoothing = time_smoothing;
    fit_data.time_smoothing_window =time_smoothing_window;
    fit_data.model_name =cur_dce_model;
    fit_data.number_rois = number_rois;
    xdata{1}.dimensions = size(currentimg);
    
    if number_rois~=0
        xdata{1}.roi_series = roi_series;
        xdata{1}.roi_series_original = roi_series_original;
        fit_data.roi_results = roi_results;
        fit_data.roi_residuals = roi_residuals;
        fit_data.roi_name = roi_name;
        
        if strcmp(cur_dce_model, 'fxr')
            xdata{1}.roi_r1 = roi_r1;
            xdata{1}.relaxivity = relaxivity;
        end
        
        if strcmp(cur_dce_model, 'auc')
            xdata{1}.roi_series_signal = roi_series_signal;
        end
        
    end
    if fit_voxels
        fit_data.fitting_results  = fitting_results;
        fit_data.voxel_residuals = voxel_residuals;
    end
    
    %     Ddata.xdata = xdata;
    %     Ddata.fit_data = fit_data;
    %     Ddata.results_a_path = results_a_path;
    %     Ddata.results_b_path = results_b_path;
    
    results{model_index} = [results_base,'.mat'];
    if save_output==true
        save(results{model_index},  'xdata','fit_data','results_b_path','hdr','-v7.3')
        %     results = fullfile(PathName, ['D_' rootname cur_dce_model '_fit_voxels.mat']);
        Opt.Input = 'file';
        % try
        %     mat_md5 = DataHash(results{model_index}, Opt);
        % catch
        %     disp('Problem using md5 hashing. Will continue');
        %     mat_md5 = 'error';
        % end
        disp(' ')
        disp('MAT results saved to: ')
        disp(results{model_index})
        % disp(['File MD5 hash: ' mat_md5])
    end
    D_vars = fit_data;
    if number_rois > 0
        D_vars.fit_parameters = D_vars.roi_results;
    end

    % Add all data needed for plot_dce_curve function
    D_vars.Cp = xdata{1}.Cp;
    D_vars.timer = xdata{1}.timer;
    if number_rois > 0
        D_vars.Ct = roi_series;
        D_vars.Ct_original = roi_series_original;
        D_vars.title = ['DCE Curve Fit - ' cur_dce_model ' ' roi_data{1}.ID];
    else
        % If no ROIs, use empty arrays
        D_vars.Ct = [];
        D_vars.Ct_original = [];
        D_vars.title = ['DCE Curve Fit - ' cur_dce_model];
    end
    D_vars.show_original = true;
    D_vars.show_ci = true;
    D_vars.x_units = 'Time (min)';
    D_vars.y_units = 'Concentration (mM)';

    % Add model-specific parameters for fxr
    if strcmp(cur_dce_model, 'fxr')
        % These would need to be calculated or passed from somewhere
        % For now, set defaults or check if they exist
        if exist('R1o', 'var')
            D_vars.R1o = R1o;
        end
        if exist('R1i', 'var')
            D_vars.R1i = R1i;
        end
        D_vars.r1 = relaxivity;
        if exist('fw', 'var')
            D_vars.fw = fw;
        end
    end
    
    % d) Check if physiologically possible, if not, remove
    %************************
    % checkind = find(x(:,1) < 0);
    % x(checkind,:) = [];
    % tumind(checkind) = [];
    
    % ve > 1
    % checkind = find(x(:,2) > 1);
    % x(checkind,:) = [];
    % tumind(checkind) = [];
    %
    % % ve < 0
    % checkind = find(x(:,2) < 0);
    % x(checkind,:) = [];
    % tumind(checkind) = [];
    %
    % % vp > 1
    % checkind = find(x(:,3) > 1);
    % x(checkind,:) = [];
    % tumind(checkind) = [];
    %
    % % vp < 0
    % checkind = find(x(:,3) < 0);
    % x(checkind,:) = [];
    % tumind(checkind) = [];
    %
    % % Ktrans > 3
    % checkind = find(x(:,1) > 3);
    % x(checkind,:) = [];
    % tumind(checkind) = [];
    %
    % % Ktrans < 0
    % checkind = find(x(:,1) < 0);
    % x(checkind,:) = [];
    % tumind(checkind) = [];
    %
    % %% d.2) Check R2 fit
    %
    % checkind = find(x(:,4) < r2filter);
    % x(checkind,:)    = [];
    % tumind(checkind) = [];
    
    
    % f) Make maps and Save image files
    %************************
    %fit_voxels = 0; %% DEBUG
    %[discard, actual] = fileparts(strrep(dynam_name, '\', '/'));
    
    if strcmp(cur_dce_model, 'tofts')
        headings = {'ROI path', 'ROI', 'Ktrans', 'Ve','SSE', 'Ktrans 95% low', ...
            'Ktrans 95% high', 'Ve 95% low', 'Ve 95% high'};
        paramname = {'Ktrans'; 've'; 'sse'; 'ktrans_ci_low'; 'ktrans_ci_high'; 've_ci_low';'ve_ci_high'};
    elseif strcmp(cur_dce_model, 'ex_tofts') || strcmp(cur_dce_model, 'nested')
        headings = {'ROI path', 'ROI', 'Ktrans', 'Ve','Vp','SSE', 'Ktrans 95% low', ...
            'Ktrans 95% high', 'Ve 95% low', 'Ve 95% high','Vp 95% low','Vp 95% high'};
        paramname = {'Ktrans'; 've'; 'vp'; 'sse'; 'ktrans_ci_low'; 'ktrans_ci_high'; 've_ci_low';'ve_ci_high'; 'vp_ci_low'; 'vp_ci_high'};  
    elseif strcmp(cur_dce_model, 'FXL_rr')
        headings = {'ROI path', 'ROI', 'Ktrans_TOI', 'Ve','Ktrans_RR','SSE', 'Ktrans_TOI 95% low', ...
            'Ktrans_TOI 95% high', 'Ve 95% low', 'Ve 95% high','Ktrans_RR 95% low','Ktrans_RR 95% high'};
        paramname = {'Ktrans_TOI'; 've'; 'Ktrans_RR'; 'sse'; 'ktransTOI_ci_low'; 'ktransTOI_ci_high'; 've_ci_low';'ve_ci_high'; 'ktransRR_ci_low'; 'ktransRR_ci_high'};
    elseif strcmp(cur_dce_model, '2cxm')
        headings = {'ROI path', 'ROI', 'Ktrans', 'Ve','Vp','Fp','SSE', 'Ktrans 95% low', ...
            'Ktrans 95% high', 'Ve 95% low', 'Ve 95% high','Vp 95% low','Vp 95% high','Fp 95% low','Fp 95% high'};
        paramname = {'Ktrans'; 've'; 'vp';'fp'; 'sse'; 'ktrans_ci_low'; 'ktrans_ci_high'; 've_ci_low';'ve_ci_high'; 'vp_ci_low'; 'vp_ci_high'; 'fp_ci_low'; 'fp_ci_high'};
    elseif strcmp(cur_dce_model, 'tissue_uptake')
        headings = {'ROI path', 'ROI', 'Ktrans', 'Fp','Vp','SSE', 'Ktrans 95% low', ...
            'Ktrans 95% high', 'Fp 95% low', 'Fp 95% high','Vp 95% low','Vp 95% high'};
        paramname = {'Ktrans'; 'fp'; 'vp'; 'sse'; 'ktrans_ci_low'; 'ktrans_ci_high'; 'fp_ci_low';'fp_ci_high'; 'vp_ci_low'; 'vp_ci_high'};
    elseif strcmp(cur_dce_model, 'patlak')
        headings = {'ROI path', 'ROI', 'Ktrans','Vp','SSE', 'Ktrans 95% low', ...
            'Ktrans 95% high','Vp 95% low','Vp 95% high'};
        paramname = {'Ktrans'; 'vp'; 'sse'; 'ktrans_ci_low'; 'ktrans_ci_high'; 'vp_ci_low'; 'vp_ci_high'};
    elseif strcmp(cur_dce_model, 'fxr')
        headings = {'ROI path', 'ROI', 'Ktrans', 'Ve','Tau','SSE', 'Ktrans 95% low', ...
            'Ktrans 95% high', 'Ve 95% low', 'Ve 95% high','Tau 95% low','Tau 95% high'};
        paramname = {'Ktrans'; 've'; 'tau'; 'sse'; 'ktrans_ci_low'; 'ktrans_ci_high'; 've_ci_low';'ve_ci_high'; 'tau_ci_low'; 'tau_ci_high'};
    elseif strcmp(cur_dce_model, 'auc')
        if quant
            headings = {'ROI path', 'ROI', 'AUC conc', 'AUC sig','NAUC conc', 'NAUC sig'};
            paramname = {'AUCc'; 'AUCs'; 'NAUCc'; 'NAUCs'};
        else
            fitting_results(:,3) = [];
            fitting_results(:,1) = [];
            
            headings = {'ROI path', 'ROI', 'AUC sig', 'NAUC sig'};
            paramname = {'AUCs'; 'NAUCs'};
        end
        
    else
        % Error
        error('Model not supported');
    end
    
    
    % Write ROI results
    if number_rois>0
        xls_results = [roi_list roi_name mat2cell(roi_results,ones(1,size(roi_results,1)),ones(1,size(roi_results,2)))];
        xls_results = [headings; xls_results];
        xls_path = [results_base '_rois.xls'];
        try
            if exist('writecell','file')
                writecell(xls_results,xls_path);
            else
                xlswrite(xls_path,xls_results);
            end
        catch exception
            disp('Problem writing XLS file. Will continue');
            xls_path
            xls_results
            disp(getReport(exception));
            xls_path='error';
        end
    end
    % Write voxel results
    if fit_voxels
        % Setup img files
        for k = 1:numel(paramname)
            curimg = zeros(size(currentimg));
            curimg(tumind) = fitting_results(:,k);
            OUTPUT(k).IMG = curimg;
            
            if outputft == 1
                % NIFTI
                nii_path{k} = [results_base '_' paramname{k} '.nii'];
                save_nii(make_nii(curimg, res, [1 1 1], [], [], hdr), nii_path{k});
            elseif outputft == 2
                % 3D DICOM
                nii_path{k} = [results_base '_' paramname{k} '.dcm'];
                curimg = OUTPUT(k).IMG;
                
                [newIMG, slope, intercept] = double2uint16Scale(curimg);
                
                if isDICOMhdr(hdr)
                    % Then we write using the dicom hdr from before
                    dicomwrite(reshape(uint16(newIMG), [size(curimg,1) size(curimg,2) 1 size(curimg,3)]),nii_path{k}, ...
                        'RescaleIntercept', intercept, 'RescaleSlope', slope, ...
                        'PixelSpacing', res(1:2)', 'SliceThickness', res(3), hdr, 'MultiframeSingleFile', 1);
                else
                    % Then we write using the dicom hdr from before
                    dicomwrite(reshape(uint16(newIMG), [size(curimg,1) size(curimg,2) 1 size(curimg,3)]),nii_path{k}, ...
                        'RescaleIntercept', intercept, 'RescaleSlope', slope, ...
                        'PixelSpacing', res(1:2)', 'SliceThickness', res(3),  'MultiframeSingleFile', 1);
                end
            elseif outputft == 3
                nii_path{k} = [results_base '_' paramname{k} '.dcm'];
                curimg = OUTPUT(k).IMG;
                
                [newIMG, slope, intercept] = double2uint16Scale(curimg);
                
                if isDICOMhdr(hdr)
                    % Then we write using the dicom hdr from before
                    dicomwrite(reshape(uint16(newIMG), [size(curimg,1) size(curimg,2) 1 size(curimg,3)]),nii_path{k}, ...
                        'ObjectType', 'MR Image Storage', 'RescaleIntercept', intercept, 'RescaleSlope', slope, ...
                        'PixelSpacing', res(1:2)', 'SliceThickness', res(3), hdr, 'MultiframeSingleFile', 0);
                else
                    % Then we write using the dicom hdr from before
                    dicomwrite(reshape(uint16(newIMG), [size(curimg,1) size(curimg,2) 1 size(curimg,3)]),nii_path{k}, ...
                        'ObjectType', 'MR Image Storage', 'RescaleIntercept', intercept, 'RescaleSlope', slope, ...
                        'PixelSpacing', res(1:2)', 'SliceThickness', res(3),  'MultiframeSingleFile', 0);
                end
            end
        end
    end
    
    
    % Calculate file hashes and log them
    Opt.Input = 'file';
    if number_rois~=0
        try
            xls_md5 = DataHash(xls_path, Opt);
        catch
            disp('Problem using md5 hashing. Will continue');
            xls_md5 = 'error';
        end
        disp(' ')
        disp('ROI results saved to: ')
        disp(xls_path)
        disp(['File MD5 hash: ' xls_md5])
    end
    if fit_voxels
        disp(' ')
        disp('Voxel results saved to: ')
        for i=1:numel(nii_path)
            try
                nii_md5 = DataHash(nii_path{i}, Opt);
            catch
                disp('Problem using md5 hashing. Will continue');
                nii_md5 = 'error';
            end
            disp(nii_path{i})
            disp(['File MD5 hash: ' nii_md5])
        end
    end
    
    disp(['Finished making maps for ' cur_dce_model '...']);
end

disp(' ');
disp('Finished D');
disp(datestr(now))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%toc
diary off;




