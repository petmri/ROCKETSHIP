
function [single_IMG, errormsg, JOB_struct, txtlog_output_path] = calculateMap(JOB_struct, dataset)

% Initialize empty variables
single_IMG = 0;
errormsg   = '';
txtlog_output_path   = '';

% Sanity check on input
if nargin < 1
    warning( 'Arguments missing' );
    return;
end

% Extract parameters

number_cpus    = JOB_struct(1).number_cpus;
neuroecon      = JOB_struct(1).neuroecon;
email          = JOB_struct(1).email;
cur_dataset    = JOB_struct(1).batch_data;
save_log       = JOB_struct(1).save_log;
email_log      = JOB_struct(1).email_log;
batch_log      = JOB_struct(1).batch_log;
current_dir    = JOB_struct(1).current_dir;
log_name       = JOB_struct(1).log_name;
submit         = JOB_struct(1).submit;
save_txt       = JOB_struct(1).save_txt;

% Extract INPUTS

file_list           = cur_dataset.file_list;
parameter_list      = cur_dataset.parameters;
parameter_list      = parameter_list(:);
fit_type            = cur_dataset.fit_type;
odd_echoes          = cur_dataset.odd_echoes;
rsquared_threshold  = cur_dataset.rsquared;
tr                  = cur_dataset.tr;
data_order          = cur_dataset.data_order;
output_basename     = cur_dataset.output_basename;
roi_list            = cur_dataset.roi_list;
fit_voxels          = cur_dataset.fit_voxels;
xy_smooth_size      = cur_dataset.xy_smooth_size;

% Add the location of the user input file if exists, else empty
if strcmp(fit_type, 'user_input')
    fit_file = cur_dataset.user_fittype_file;
    ncoeffs  = cur_dataset.ncoeffs;
    coeffs   = cur_dataset.coeffs;
    tr_present=cur_dataset.tr_present;
else
    fit_file = '';
    ncoeffs  = 0;
    coeffs   = '';
    tr_present='';
end

% Start logging txt if save_txt
if save_txt && submit
    imagefile=cell2mat(file_list(end));
    % Read file and get header information
    [file_path, filename]  = fileparts(imagefile);
    
    if exist(file_path)
        % Make log name the same as the saved map name
        txtlog_output_path = fullfile(file_path, [output_basename, '_', fit_type,'_', filename ...
                ,'.txt']);
        diary(txtlog_output_path);
    else
        errormsg = warning('Path of files does not exist');
        return;
    end
end


%------------------------------------
% file_list = {'file1';'file2'};
% 					% must point to valid nifti files
% parameter_list = [10.5 21 31.5 42 52.5 63]';
% 					% units of ms or degrees
% fit_type = 'linear_weighted';
% 					% options{'none','t2_linear_simple','t2_linear_weighted','t2_exponential','t2_linear_fast'
% 					%			't1_tr_fit','t1_fa_fit','t1_fa_linear_fit','t1_ti_exponential_fit'}
% odd_echoes = 0;	% boolean, if selected only odd parameters will be
% 					% used for fit
% rsquared_threshold = 0.2;
% 					% all fits with R^2 less than this set to -1
% number_cpus = 4;	% not used if running on neuroecon
% neuroecon = 0;	% boolean
% output_basename = 'foo';
% 					% base of output filename
% data_order = 'xyzn';% in what order is the data organized
% 					% options{'xynz','xyzn','xyzfile'}
% tr = 20;			% units ms, only used for T1 FA fitting
% submit            % Let's the function know if this is a tester job of
%                     actual file generation

%------------------------------------



ok_ = isfinite(parameter_list) & ~isnan(parameter_list);
if ~all( ok_ ) || isempty(parameter_list)
    errormsg = warning( 'TE/TR/FA/TI list contains invalid values' );
    disp(parameter_list);
    return;
end

for m=size(file_list,1):-1:1
    testfile=cell2mat(file_list(m));
    if ~exist(testfile, 'file')
        % File does not exist.
        errormsg = warning( 'File does not exist' );
        disp(testfile);
        return;
    end
    for n=(m-1):-1:1
        comparefile=file_list(n);
        if strcmp(testfile,comparefile)
            disp( 'Removing duplicates' );
            disp(comparefile);
            file_list(m)=[];
        end
    end
end

if submit
    % fit_type = 'none';
    disp(['Starting execution at ', datestr(now,'mmmm dd, yyyy HH:MM:SS')])
    disp('User selected files: ');
    [nrows,ncols]= size(file_list);
    for row=1:nrows
        disp(file_list{row,:})
    end
    disp('User selected TE/TR/FA/TI: ');
    disp(parameter_list);
    disp('User selected fit: ');
    disp(fit_type);
    disp('User selected CPUs: ');
    disp(number_cpus);
    disp('User selected Neuroecon: ');
    disp(neuroecon);
    disp('User selected email: ');
    disp(email);
    disp('User selected data order: ');
    disp(data_order);
    disp('User selected output basename: ');
    disp(output_basename);
    disp('User selected only odd echoes: ');
    disp(odd_echoes);
    disp('User selected smooth size (Gaussian standard deviation): ');
    disp(xy_smooth_size);
    disp('User selected fit all voxels: ');
    disp(fit_voxels);
    disp('User selected ROIs: ');
    [nrows,ncols]= size(roi_list);
    for row=1:nrows
        disp(roi_list{row,:})
    end
    if ~isempty(strfind(fit_type,'t1')) && ~isempty(strfind(fit_type,'fa'))
        disp('User selected tr: ');
        disp(tr);
    end
end
disp('User selected r^2 threshold: ');
disp(rsquared_threshold);

if strcmp(fit_type, 'user_input')
    disp('User selected fit file: ');
    disp(fit_file);
end


% return;

% Calculate number of fits

dim_n = numel(parameter_list);
if strcmp(data_order,'xyzfile')
    number_of_fits = size(file_list,1)/dim_n;
    if rem(size(file_list,1),dim_n)~=0
        warning on
        errormsg = warning( 'Number of files not evenly divisible by number or parameters' );
        return;
    end
else
    number_of_fits = size(file_list,1);
end

% Auto detect GPU
try
    gpufit_available = GpufitCudaAvailableMex;
catch
    gpufit_available = 0;
end

if gpufit_available
    disp("Gpufit detected. GPU will be utilized for T1 mapping.")
else
    disp("Gpufit detection failed. Defaulting to CPU.")
end

para_prefs = parse_preference_file('parametric_preferences.txt',0,{'force_cpu'},{0});
if str2num(para_prefs.force_cpu)
    gpufit_available = 0;
end

% Create parallel processing pool
try
    if ~neuroecon && ~gpufit_available && ~isempty(ver('parallel'))
        r_prefs = parse_preference_file('parametric_preferences.txt',0,{'use_matlabpool'},{0});
        if str2num(r_prefs.use_matlabpool)
            s = matlabpool('size');
            if s~=number_cpus
                if s>0
                    matlabpool close
                end
                matlabpool('local', number_cpus); % Check
            end
            if strcmp(fit_type, 'user_input') && submit
                matlabpool('ADDATTACHEDFILES', {fit_file});
            end  
        else
            s = gcp('nocreate');
            if isempty(s)
                parpool('local',number_cpus);
            else
                if s.NumWorkers~=number_cpus
                    delete(gcp('nocreate'))
                    parpool('local',number_cpus);
                end
            end
            if strcmp(fit_type, 'user_input') && submit
               parpool('ADDATTACHEDFILES', {fit_file});
            end    
        end
    end
catch
    warning('Parallel pooling failed. This may be due to a licensing issue or lack of installation.')
end

execution_time = zeros(size(file_list,1),1);

% do processing
for n=1:number_of_fits
    tic %start timer
    
    % Read only one file then process it
    if ~strcmp(data_order,'xyzfile')
        imagefile=cell2mat(file_list(n));
        
        % Read file and get header information
        [file_path, filename ext]  = fileparts(imagefile);
        nii = load_untouch_nii(imagefile);
        res = nii.hdr.dime.pixdim;
        res = res(2:4);
        image_3d = nii.img;
        input_hdr = nii.hdr;
        [dim_x, dim_y, dim_zn] = size(image_3d);
        % 		dim_n = size(parameter_list,1);
        dim_z = dim_zn / dim_n;
        
        % Reshape image to extract individual decay curves
        % shaped to be four dimensional with dimensions [x,y,z,te]
        if strcmp(data_order,'xynz')
            shaped_image = reshape(image_3d,dim_x,dim_y,dim_n,dim_z);
            shaped_image = permute(shaped_image,[1,2,4,3]);
        elseif strcmp(data_order,'xyzn')
            shaped_image = reshape(image_3d,dim_x,dim_y,dim_z,dim_n);
            % 			shaped_image = permute(shaped_image,[1,2,4,3]);
        else
            warning on
            warning( 'Unknown data order' );
            errormsg = warning( 'Number of files not evenly divisible by number or parameters' );
            return;
        end
        
        % Read all files as all are needed for fit
    else
        % For each file in list load and add to larger matrix
        for m=1:dim_n
            imagefile=cell2mat(file_list(m+(n-1)*dim_n));
            
            % Read file and get header information
            [file_path, filename]  = fileparts(imagefile);
            nii = load_untouch_nii(imagefile);
            res = nii.hdr.dime.pixdim;
            res = res(2:4);
            image_3d = nii.img;
            input_hdr = nii.hdr;
            
            % Resize to small for visualization
            if ~submit
                for j = 1:size(image_3d,3)
                    image_3d_small(:,:,j) = imresize(image_3d(:,:,j), 0.4);
                end
                image_3d = image_3d_small;
            end
            
            [dim_x, dim_y, dim_z] = size(image_3d);
            dim_n = size(parameter_list,1);
            
            if m==1
                shaped_image = zeros([dim_x dim_y dim_z dim_n]);
            end
            shaped_image(:,:,:,m) = image_3d;
        end
    end
    
    % Remove even echoes if requested
    if odd_echoes
        dim_n = floor((dim_n+1)/2);
        temp_image = zeros(dim_x,dim_y,dim_z,dim_n);
        temp_n = zeros(dim_n,1);
        for m=1:dim_n
            temp_image(:,:,:,m) = shaped_image(:,:,:,1+2*(m-1));
            temp_n(m) = parameter_list(1+2*(m-1));
        end
        shaped_image = temp_image;
        parameter_list = temp_n;
    end
    
    % Smooth XY if requested
    if xy_smooth_size~=0 && fit_voxels
        % make size 3*sigma rounded to nearest odd
        kernel_size_odd = 2.*round((xy_smooth_size*3+1)/2)-1;
        if kernel_size_odd<3
            kernel_size_odd = 3;
        end
        h = fspecial('gaussian', [kernel_size_odd kernel_size_odd],xy_smooth_size);
        for i=1:dim_z
            for j=1:dim_n
                single_image = shaped_image(:,:,i,j);
                smooth_image = filter2(h, single_image);
                shaped_image(:,:,i,j) = smooth_image;
            end
        end
    end
    
    % Change fittype to linear if needed for visualization
    if ~submit
        if ~isempty(strfind(fit_type,'t2'))
            fit_type = 't2_linear_fast';
        elseif ~isempty(strfind(fit_type,'t1_fa'))
            fit_type = 't1_fa_linear_fit';
        elseif ~isempty(strfind(fit_type, 'ADC'))
            fit_type = 'ADC_linear_fast';
        elseif ~isempty(strfind(fit_type, 'user_input'))
            fit_type = 'user_input';
        else
           
        end
    end
    
    
    % Perpare any ROIs
    number_rois = 0;
    if ~isempty(roi_list) && submit
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
        if number_rois~=0
            [~, roi_name, roi_ext] = arrayfun(@(x) fileparts(x{:}), roi_list, 'UniformOutput', false);
            
            %Load ROI, find the selected voxels
            for r=number_rois:-1:1
                single_file=cell2mat(roi_list(r));
                
                if strcmp(roi_ext(r),'.nii') || strcmp(roi_ext(r),'.hdr') || strcmp(roi_ext(r),'.img')
                    single_roi = load_untouch_nii(single_file);
                    single_roi = double(single_roi.img);
                    roi_index{r}= find(single_roi > 0);
                elseif strcmp(roi_ext(r),'.roi')
                    single_roi = ReadImageJROI(single_file);
                    if strcmp(single_roi.strType,'Polygon') || strcmp(single_roi.strType,'Freehand')
                        roi_image = poly2mask(...
                            single_roi.mnCoordinates(:,2)+0.5,...
                            single_roi.mnCoordinates(:,1)+0.5,...
                            dim_x,dim_y);
                        roi_index{r}= find(roi_image > 0);
                        
                    elseif strcmp(single_roi.strType,'Rectangle')
                        roi_image = poly2mask(...
                            [single_roi.vnRectBounds(1:2:3) fliplr(single_roi.vnRectBounds(1:2:3))]+0.5,...
                            [single_roi.vnRectBounds(2) single_roi.vnRectBounds(2) single_roi.vnRectBounds(4) single_roi.vnRectBounds(4)]+0.5,...
                            dim_x,dim_y);
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
                        
                        roi_image = poly2mask(X,Y,dim_x,dim_y);
                        
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
                        roi_index{r} = roi_index{r} + z_pos*dim_x*dim_y;
                    else
                        error( 'Could not find Z position in ROI file, use latest ImageJ verion' );
                    end
                else
                    warning( 'File type for ROI not supported' );
                    disp(roi_ext(r));
                    return;
                end
            end
            

%             original_timepoint = zeros(dim_x,dim_y,dim_z);
            roi_series = zeros(number_rois,1,1,dim_n);
            for t=1:dim_n
                original_timepoint = shaped_image(:,:,:,t);

                %Average ROI voxels, insert into time series
                for r=number_rois:-1:1
                    roi_series(r,1,1,t) = mean(original_timepoint(roi_index{r}));
                    
%                     original_timepoint(roi_index{r}) = max(max(max(original_timepoint)));
                end
            end
            
%             figure
%             imshow3D(original_timepoint, []);
            
            %make backup
            roi_series_original = roi_series;
        end
    end
    
    if ~fit_voxels && number_rois==0
        disp('nothing to fit, select an ROI file or check "fit voxels"');
        return;
    end
    
    
    % Run Fitting Algorithms Voxels
    if(fit_voxels)
       
        % Fit the model
        if(gpufit_available && strcmp(fit_type, 't1_fa_fit'))
            disp("GPU detected, using T1 exponential GPU fitting")
            model_id = ModelID.T1_FA_EXPONENTIAL;
            estimator_id = EstimatorID.LSE;
            
            tolerance = 1e-12;
            max_n_iterations = 200;
    
            [dim_x, dim_y, dim_z, dim_te] = size(shaped_image);
            linear_shape = reshape(shaped_image,dim_x*dim_y*dim_z,dim_te);
            number_voxels = dim_x*dim_y*dim_z;
            si = linear_shape';
            si = cast(si,'double');
            
            % cope scaling to help convergence
%             si_max = max(si,[],"all");
%             si_max = reshape(max(shaped_image,[],4)*10, dim_x*dim_y*dim_z,[])';
%             if si_max > 0
%                 si = si./si_max;
%             end
            
            init_param = zeros([2,number_voxels]);
            init_param(1,:) = reshape(max(shaped_image,[],4)*10, dim_x*dim_y*dim_z,[])';
            for i=1:number_voxels
%                 index_i = i * dim_te;
%                 si_i = si(index_i:index_i+dim_te);
%                 init_param(1,i) = max(si_i)*10;
%                 init_param(1,i) = 1;
                init_param(2,i) = 500;
            end
            init_param_single = single(init_param);
            
            constraints = zeros([4,number_voxels]);
            for i=1:number_voxels
                constraints(1,i) = 0;
                constraints(2,i) = inf;
                constraints(3,i) = 0;
                constraints(4,i) = 10000;
            end
            constraints_single = single(constraints);
            % constrain upper and lower bounds for both parameters
            constraint_types = int32([3,3]);
            
            % Load measured data
            tr_array = tr*ones(size(parameter_list));
            indie_vars = single([parameter_list' tr_array']);
            si_single = single(si);

            % Execute GPU fit
            [parameters, states, chi_squares, n_iterations, time] = ...
                gpufit_constrained(si_single,[], model_id, init_param_single, ...
                constraints_single, constraint_types, tolerance, max_n_iterations, ...
                [], estimator_id, indie_vars);
            
            % convergence stats
%             size(states(states==1))/number_voxels*100
%             size(states(states==2))/number_voxels*100

            % If did not converge discard values
            one_parameter = parameters(1,:);
            one_parameter(states~=0) = NaN;  %a
            parameters(1,:) = one_parameter;
            one_parameter = parameters(2,:);
            one_parameter(states~=0) = NaN;  %t1
            parameters(2,:) = one_parameter;

            fit_output(:,1) = parameters(2,:);
            fit_output(:,2) = parameters(1,:);
            fit_output(:,3) = chi_squares;
            fit_output(:,4) = zeros(number_voxels,1);
            fit_output(:,5) = zeros(number_voxels,1);
            fit_output(:,6) = zeros(number_voxels,1);
            
    %         for i=1:number_voxels
    %             % filter negatives
    %             if parameters(1,i) > 0
    %                 GG(i,1) = parameters(1,i); %Ktrans
    %             end
    %             GG(i,2) = parameters(2,i); %vp

        elseif (neuroecon)
            %Schedule object, neuroecon
            sched = findResource('scheduler', 'configuration', 'NeuroEcon.local');
            set(sched, 'SubmitArguments', '-l walltime=12:00:00 -m abe -M thomasn@caltech.edu')

            warning off; %#ok<WNOFF>

            p = pwd;
            %         n = '/home/thomasn/scripts/niftitools';

            job = createMatlabPoolJob(sched, 'configuration', 'NeuroEcon.local','PathDependencies', {p});
            set(job, 'MaximumNumberOfWorkers', 20);
            set(job, 'MinimumNumberOfWorkers', 1);
            createTask(job, @parallelFit, 1,{parameter_list,fit_type,shaped_image,tr, submit, fit_file, ncoeffs, coeffs, tr_present,rsquared_threshold});

            submit(job);
            waitForState(job)
            results = getAllOutputArguments(job);
            destroy(job);

            fit_output = cell2mat(results);
        else
            fit_output = parallelFit(parameter_list,fit_type,shaped_image,tr, submit, fit_file, ncoeffs, coeffs, tr_present,rsquared_threshold);
        end
    end
    
    % Run Fitting Algorithms ROIs
    if(number_rois)
        % Run
        if(neuroecon)
            %Schedule object, neuroecon
            sched = findResource('scheduler', 'configuration', 'NeuroEcon.local');
            set(sched, 'SubmitArguments', '-l walltime=12:00:00 -m abe -M thomasn@caltech.edu')

            warning off; %#ok<WNOFF>

            p = pwd;
            %         n = '/home/thomasn/scripts/niftitools';

            job = createMatlabPoolJob(sched, 'configuration', 'NeuroEcon.local','PathDependencies', {p});
            set(job, 'MaximumNumberOfWorkers', 20);
            set(job, 'MinimumNumberOfWorkers', 1);
            createTask(job, @parallelFit, 1,{parameter_list,fit_type,roi_series,tr, submit, fit_file, ncoeffs, coeffs, tr_present,rsquared_threshold});

            submit(job);
            waitForState(job)
            results = getAllOutputArguments(job);
            destroy(job);

            roi_output = cell2mat(results);
        else
            roi_output = parallelFit(parameter_list,fit_type,roi_series,tr, submit, fit_file, ncoeffs, coeffs, tr_present,rsquared_threshold);
        end
    end
    
    if fit_voxels
        if strfind(fit_type, 'user_input') 
            for i = 1:ncoeffs
                eval([coeffs{i} '_fit = fit_output(:,' num2str(i) ');']);
                eval([coeffs{i} '_cilow = fit_output(:,' num2str(ncoeffs+i+1) ');']);
                eval([coeffs{i} '_cihigh= fit_output(:,' num2str(ncoeffs+i+1) ');']);
            end

            r_squared				 = fit_output(:,ncoeffs+1);

            % Throw out bad results
            ind = [];
            for i = 1:ncoeffs
                ind = [ind, eval(['find(' coeffs{i} '_fit ~= -2);'])];
                ind = unique(ind);
            end

            indr = find(r_squared < rsquared_threshold);

            indbad = intersect(indr, ind);

            for i = 1:ncoeffs
                eval([coeffs{i} '_fit(indbad) = -1;']);
                eval([coeffs{i} '_fit = reshape(' coeffs{i} '_fit, [dim_x, dim_y, dim_z]);']);

                eval([coeffs{i} '_cilow(indbad) = -1;']);
                eval([coeffs{i} '_cilow = reshape(' coeffs{i} '_cilow, [dim_x, dim_y, dim_z]);']);

                eval([coeffs{i} '_cihigh(indbad) = -1;']);
                eval([coeffs{i} '_cihigh = reshape(' coeffs{i} '_cihigh, [dim_x, dim_y, dim_z]);']);
            end

            r_squared				= reshape(r_squared, [dim_x, dim_y, dim_z]);

        else
            % Collect and reshape outputs
            exponential_fit			 = fit_output(:,1);
            rho_fit					 = fit_output(:,2);
            r_squared				 = fit_output(:,3);
            confidence_interval_low	 = fit_output(:,4);
            confidence_interval_high = fit_output(:,5);
            
            % Throw out bad results
            if (gpufit_available)
                % Gpufit uses chi squared
                indr = find(r_squared > inf);
            else
                indr = find(r_squared < rsquared_threshold);
            end
            inde = find(exponential_fit ~=-2);
            m    =intersect(indr,inde);
            
            rho_fit(m) = -1;
            exponential_fit(m) = -1;
            confidence_interval_low(m) = -1;
            confidence_interval_high(m) = -1;

            exponential_fit			= reshape(exponential_fit, [dim_x, dim_y, dim_z]);
            rho_fit					= reshape(rho_fit,  [dim_x, dim_y, dim_z]); %#ok<NASGU>
            r_squared				= reshape(r_squared, [dim_x, dim_y, dim_z]);
            confidence_interval_low  = reshape(confidence_interval_low, [dim_x, dim_y, dim_z]);
            confidence_interval_high = reshape(confidence_interval_high, [dim_x, dim_y, dim_z]);
        end

        if submit
            if strfind(fit_type, 'user_input')
                % Create output names
                for i = 1:ncoeffs
                    fullpathT2{i} = fullfile(file_path, [output_basename, '_', coeffs{i},'_', filename ...
                        ,'.nii']);
                    fullpathCILow{i}   = fullfile(file_path, ['CI_low_', coeffs{i},'_', filename ...
                        , '.nii']);
                    fullpathCIHigh{i}   = fullfile(file_path, ['CI_high_', coeffs{i},'_', filename ...
                        , '.nii']);
                end
                fullpathRsquared   = fullfile(file_path, ['Rsquared_', fit_type,'_', filename ...
                    , '.nii']);

                % Write output
                for i = 1:ncoeffs
                    eval(['T2dirnii(' num2str(i) ').nii = make_nii(' coeffs{i} '_fit, res, [1 1 1], [], output_basename,input_hdr);']);
                    eval(['save_nii(T2dirnii(' num2str(i) ').nii, fullpathT2{' num2str(i) '});']);

                    eval(['CILOWdirnii(' num2str(i) ').nii = make_nii(' coeffs{i} '_cilow, res, [1 1 1], [], output_basename,input_hdr);']);
                    eval(['save_nii(CILOWdirnii(' num2str(i) ').nii, fullpathCILow{' num2str(i) '});']);

                    eval(['CIHIGHdirnii(' num2str(i) ').nii = make_nii(' coeffs{i} '_cihigh, res, [1 1 1], [], output_basename,input_hdr);']);
                    eval(['save_nii(CIHIGHdirnii(' num2str(i) ').nii, fullpathCIHigh{' num2str(i) '});']);
                end

                Rsquareddirnii   = make_nii(r_squared, res, [1 1 1], [], 'R Squared of fit',input_hdr);
                save_nii(Rsquareddirnii, fullpathRsquared);

            else
                % Create output names
                fullpathT2 = fullfile(file_path, [output_basename, '_', fit_type,'_', filename ...
                    ,'.nii']);
                fullpathRsquared   = fullfile(file_path, ['Rsquared_', fit_type,'_', filename ...
                    , '.nii']);
                fullpathCILow   = fullfile(file_path, ['CI_low_', fit_type,'_', filename ...
                    , '.nii']);
                fullpathCIHigh   = fullfile(file_path, ['CI_high_', fit_type,'_', filename ...
                    , '.nii']);

                % Write output
                T2dirnii = make_nii(exponential_fit, res, [1 1 1], [], fit_type,input_hdr);
                Rsquareddirnii   = make_nii(r_squared, res, [1 1 1], [], 'R Squared of fit',input_hdr);
                save_nii(T2dirnii, fullpathT2);
                save_nii(Rsquareddirnii, fullpathRsquared);
                % Linear_fast does not calculate confidence intervals
                if ~strcmp(fit_type,'t2_linear_fast') && ~strcmp(fit_type,'t1_fa_linear_fit')
                    CILowdirnii  = make_nii(confidence_interval_low, res, [1 1 1], [], 'Low 95% confidence interval',input_hdr);
                    CIHighdirnii  = make_nii(confidence_interval_high, res, [1 1 1], [], 'High 95% confidence interval',input_hdr);
                    save_nii(CILowdirnii, fullpathCILow);
                    save_nii(CIHighdirnii, fullpathCIHigh);
                end
            end

            disp(['Map completed at ', datestr(now,'mmmm dd, yyyy HH:MM:SS')])
            disp('Map saved to: ');
            if iscell(fullpathT2)
                for i = 1:numel(fullpathT2)
                    disp(fullpathT2{i});
                end
            else
                disp(fullpathT2);
            end

            single_IMG = 1;
        else
            % Process R2 map for quick visualization
            single_IMG = r_squared;
        end
    end
    if number_rois
        if strcmp(fit_type,'t2_exponential_plus_c')
            headings = {'ROI path', 'ROI', fit_type, 'rho', 'r squared', '95% CI low', '95% CI high', 'Sum Squared Error', 'C', 'C 95% CI low', 'C 95% CI high'};
        else
            headings = {'ROI path', 'ROI', fit_type, 'rho', 'r squared', '95% CI low', '95% CI high', 'Sum Squared Error'};
        end
        xls_results = [roi_list roi_name mat2cell(roi_output,ones(1,size(roi_output,1)),ones(1,size(roi_output,2)))];
        xls_results = [headings; xls_results];
        xls_path = fullfile(file_path, [output_basename, '_', fit_type,'_', filename,'.xls']);
        xlswrite(xls_path,xls_results);

        disp('ROI saved to: ');
        disp(xls_path);
        single_IMG = 1;
    end
    
    if submit
        execution_time(n) = toc;
        disp(['Execution time was: ',datestr(datenum(0,0,0,0,0,execution_time(n)),'HH:MM:SS')]);
    end
end

% Close parallel processing pool
% if ~neuroecon && exist('matlabpool') && submit
%     matlabpool close;
% end

if submit
    total_time = sum(execution_time);
    disp(['All processing complete at ', datestr(now,'mmmm dd, yyyy HH:MM:SS')])
    disp(['Total execution time was: ',datestr(datenum(0,0,0,0,0,total_time),'HH:MM:SS')]);
    
    % The map was calculated correctly, so we note this in the data structure
    cur_dataset.to_do = 0;
    JOB_struct(1).batch_data = cur_dataset;
    
    
    % Create data structures for curve fit analysis
    fit_data.model_name = fit_type;
    fit_data.fit_file = fit_file;
    fit_data.ncoeffs = ncoeffs;
    fit_data.coeffs = coeffs;
    fit_data.number_rois = number_rois;
    fit_data.fit_voxels = 0; %Reset below if fit_voxels
    xdata{1}.tr = tr;
    xdata{1}.dimensions = [dim_x, dim_y, dim_z];
    xdata{1}.numvoxels = 0; %Reset below if fit_voxels
    xdata{1}.x_values = parameter_list;
    
    
    if strfind(fit_type,'ADC')
        xdata{1}.x_units = 'b-value (s/mm^2)';
    elseif strfind(fit_type,'fa')
        xdata{1}.x_units = 'FA (degrees)';
    elseif strcmp(fit_type,'user_input')
        xdata{1}.x_units = 'a.u.';
    elseif strfind(fit_type,'t1')
        xdata{1}.x_units = 'TR (ms)';
    elseif strfind(fit_type,'t2')
        xdata{1}.x_units = 'TE (ms)';
    else
        xdata{1}.x_units = 'ms';
    end
    xdata{1}.y_units = 'a.u.';    
    if fit_voxels
        fit_data.fit_voxels = logical(numel(shaped_image));
        fit_data.fitting_results = fit_output;
        
        xdata{1}.y_values = shaped_image;
        xdata{1}.numvoxels = numel(fit_output);
        
        local_log_name = strrep(fullpathT2, '.nii', '.mat');
    end
    if number_rois
        fit_data.roi_results = roi_output;
%         fit_data.roi_residuals = roi_residuals;
        fit_data.roi_name = roi_name;
        
        xdata{1}.roi_y_values = roi_series;
        
        if ~exist('local_log_name','var')
            local_log_name = strrep(xls_path, '.xls', '.mat');
        end
    end
    
    % Save text and data structure logs if desired.
    if save_log
        save(local_log_name, 'JOB_struct','fit_data','xdata', '-mat');
        disp(['Saved log at: ' local_log_name]);
    end
    
    
    
end

if submit
    diary off
end




