function [GG, residuals] = FXLfit_generic(xdata, number_voxels, model, verbose, cpu_only)

if nargin < 5
    cpu_only = 0;
    if nargin < 4
        verbose = 0;
        cpu_only = 0;
    end
end
p = 0;
residuals = [];

% check (again) if using gpu
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
gpu_prefs = parse_preference_file('dce_preferences.txt',0,{'force_cpu'},{0});
FORCE_CPU = str2num(gpu_prefs.force_cpu);
if FORCE_CPU
    disp("Forcing CPU for voxel fitting...")
    USE_GPU = 0;
end


if strcmp(model, 'ex_tofts')
    % Slice out needed variables for speed
    Ct_data = xdata{1}.Ct;
    Cp_data = xdata{1}.Cp;
    timer_data = xdata{1}.timer;
    % Get values from pref file
    prefs_str = parse_preference_file('dce_preferences.txt',0,...
        {'voxel_lower_limit_ktrans' 'voxel_upper_limit_ktrans' 'voxel_initial_value_ktrans' ...
        'voxel_lower_limit_ve' 'voxel_upper_limit_ve' 'voxel_initial_value_ve' ...
        'voxel_lower_limit_vp' 'voxel_upper_limit_vp' 'voxel_initial_value_vp'});
    prefs.lower_limit_ktrans = str2num(prefs_str.voxel_lower_limit_ktrans);
    prefs.upper_limit_ktrans = str2num(prefs_str.voxel_upper_limit_ktrans);
    prefs.initial_value_ktrans = str2num(prefs_str.voxel_initial_value_ktrans);
    prefs.lower_limit_ve = str2num(prefs_str.voxel_lower_limit_ve);
    prefs.upper_limit_ve = str2num(prefs_str.voxel_upper_limit_ve);
    prefs.initial_value_ve = str2num(prefs_str.voxel_initial_value_ve);
    prefs.lower_limit_vp = str2num(prefs_str.voxel_lower_limit_vp);
    prefs.upper_limit_vp = str2num(prefs_str.voxel_upper_limit_vp);
    prefs.initial_value_vp = str2num(prefs_str.voxel_initial_value_vp);
    %Log values used
    if verbose
        fprintf('lower_limit_ktrans = %s\n',num2str(prefs.lower_limit_ktrans));
        fprintf('upper_limit_ktrans = %s\n',num2str(prefs.upper_limit_ktrans));
        fprintf('initial_value_ktrans = %s\n',num2str(prefs.initial_value_ktrans));
        fprintf('lower_limit_ve = %s\n',num2str(prefs.lower_limit_ve));
        fprintf('upper_limit_ve = %s\n',num2str(prefs.upper_limit_ve));
        fprintf('initial_value_ve = %s\n',num2str(prefs.initial_value_ve));
        fprintf('lower_limit_vp = %s\n',num2str(prefs.lower_limit_vp));
        fprintf('upper_limit_vp = %s\n',num2str(prefs.upper_limit_vp));
        fprintf('initial_value_vp = %s\n',num2str(prefs.initial_value_vp));
    end
        
    if ~USE_GPU || cpu_only
        % Get values from pref file
        prefs_str = parse_preference_file('dce_preferences.txt',0,...
            {'voxel_TolFun' 'voxel_TolX' 'voxel_MaxIter' 'voxel_MaxFunEvals' 'voxel_Robust'});
        prefs.TolFun = str2num(prefs_str.voxel_TolFun);
        prefs.TolX = str2num(prefs_str.voxel_TolX);
        prefs.MaxIter = str2num(prefs_str.voxel_MaxIter);
        prefs.MaxFunEvals = str2num(prefs_str.voxel_MaxFunEvals);
        prefs.Robust = prefs_str.voxel_Robust;
        %Log values used
        if verbose
            fprintf('TolFun = %s\n',num2str(prefs.TolFun));
            fprintf('TolX = %s\n',num2str(prefs.TolX));
            fprintf('MaxIter = %s\n',num2str(prefs.MaxIter));
            fprintf('MaxFunEvals = %s\n',num2str(prefs.MaxFunEvals));
            fprintf('Robust = %s\n',num2str(prefs.Robust));
        end

        % Preallocate for speed
        GG = zeros([number_voxels 10],'double');
        residuals = zeros([number_voxels numel(xdata{1}.timer)],'double');
        %Turn off diary if on as it doesn't work with progress bar
        diary_restore = 0;
        if strcmp(get(0,'Diary'),'on')
            diary off;
            diary_restore = 1;
        end
        if verbose; p = ProgressBar(number_voxels,'verbose',verbose); end
    
        parfor i = 1:number_voxels
            [GG(i,:), residuals(i,:)] = model_extended_tofts(Ct_data(:,i),Cp_data,timer_data,prefs);
            if verbose; p.progress; end
        end
    
        if verbose; p.stop; end
        if diary_restore, diary on, end
    else        
        model_id = ModelID.TOFTS_EXTENDED;
        estimator_id = EstimatorID.LSE;
        
        % Load GPU fitting values from pref file
        prefs_str = parse_preference_file('dce_preferences.txt',0,...
            {'gpu_tolerance' 'gpu_max_n_iterations'});
        prefs.gpu_tolerance = str2num(prefs_str.gpu_tolerance);
        prefs.gpu_max_n_iterations = str2num(prefs_str.gpu_max_n_iterations);
        %Log values used
        if verbose
            fprintf('GPU Max iterationos = %s\n',num2str(prefs.gpu_max_n_iterations));
            fprintf('GPU Tolerance = %s\n',num2str(prefs.gpu_tolerance));
        end   
        
        tolerance = prefs.gpu_tolerance;
        max_n_iterations = prefs.gpu_max_n_iterations;
        
        init_param = zeros([3,number_voxels]);
        for i=1:number_voxels
            init_param(1,i) = prefs.initial_value_ktrans;
            init_param(2,i) = prefs.initial_value_ve;
            init_param(3,i) = prefs.initial_value_vp;
        end
        init_param_single = single(init_param);
        
        constraints = zeros([6,number_voxels]);
        constraint_type = zeros([3,number_voxels],'int32');
        for i=1:number_voxels
            constraints(1,i) = prefs.lower_limit_ktrans;
            constraints(2,i) = prefs.upper_limit_ktrans;
            constraints(3,i) = prefs.lower_limit_ve;
            constraints(4,i) = prefs.upper_limit_ve;
            constraints(5,i) = prefs.lower_limit_vp;
            constraints(6,i) = prefs.upper_limit_vp;
            constraint_type(1,i) = 3;
            constraint_type(2,i) = 3;
            constraint_type(3,i) = 3;
        end
        constraints_single = single(constraints);
        
        % Load measured data
        indie_vars = single([timer_data' Cp_data]);
        Ct_single = single(Ct_data);
        
        % Execute GPU fit
        %[parameters, states, chi_squares, n_iterations, time] = gpufit(Ct_single,[],model_id,init_param_single,tolerance, max_n_iterations,[],estimator_id,indie_vars);
        [parameters, states, chi_squares, n_iterations, time] = gpufit_constrained(Ct_single,[],model_id,init_param_single,constraints_single,constraint_type,tolerance, max_n_iterations,[],estimator_id,indie_vars);
        %fprintf('GPU fit finished in %s seconds\n',num2str(time));
        
        % If did not converge discard values
        one_parameter = parameters(1,:);
        one_parameter(states~=0) = NaN;  %Ktrans
        parameters(1,:) = one_parameter;
        one_parameter = parameters(2,:);
        one_parameter(states~=0) = NaN;  %ve
        parameters(2,:) = one_parameter;
        one_parameter = parameters(3,:);
        one_parameter(states~=0) = NaN;  %vp
        parameters(3,:) = one_parameter;
        
        GG = [parameters' chi_squares'];
        % add zeros for the unknown + and - 95 CI
        GG = [GG zeros(number_voxels, 6)];
        residuals = [];
        
%         for i=1:number_voxels
%             % filter negatives
%             if parameters(1,i) > 0
%                 GG(i,1) = parameters(1,i); %Ktrans
%             end
%             GG(i,2) = parameters(2,i); %vp
%             GG(i,3) = parameters(3,i);
%         end
    end
elseif strcmp(model, 'tissue_uptake')
    % Slice out needed variables for speed
    Ct_data = xdata{1}.Ct;
    Cp_data = xdata{1}.Cp;
    timer_data = xdata{1}.timer;
    % Get values from pref file
    prefs_str = parse_preference_file('dce_preferences.txt',0,...
        {'voxel_lower_limit_ktrans' 'voxel_upper_limit_ktrans' 'voxel_initial_value_ktrans' ...
        'voxel_lower_limit_fp' 'voxel_upper_limit_fp' 'voxel_initial_value_fp' ...
        'voxel_lower_limit_tp' 'voxel_upper_limit_tp' 'voxel_initial_value_tp' ...
        'voxel_lower_limit_vp' 'voxel_upper_limit_vp' 'voxel_initial_value_vp'});
    prefs.lower_limit_ktrans = str2num(prefs_str.voxel_lower_limit_ktrans);
    prefs.upper_limit_ktrans = str2num(prefs_str.voxel_upper_limit_ktrans);
    prefs.initial_value_ktrans = str2num(prefs_str.voxel_initial_value_ktrans);
    prefs.lower_limit_fp = str2num(prefs_str.voxel_lower_limit_fp);
    prefs.upper_limit_fp = str2num(prefs_str.voxel_upper_limit_fp);
    prefs.initial_value_fp = str2num(prefs_str.voxel_initial_value_fp);
    prefs.lower_limit_tp = str2num(prefs_str.voxel_lower_limit_tp);
    prefs.upper_limit_tp = str2num(prefs_str.voxel_upper_limit_tp);
    prefs.initial_value_tp = str2num(prefs_str.voxel_initial_value_tp);
    prefs.lower_limit_vp = str2num(prefs_str.voxel_lower_limit_vp);
    prefs.upper_limit_vp = str2num(prefs_str.voxel_upper_limit_vp);
    prefs.initial_value_vp = str2num(prefs_str.voxel_initial_value_vp);
    %Log values used
    if verbose
        fprintf('lower_limit_ktrans = %s\n',num2str(prefs.lower_limit_ktrans));
        fprintf('upper_limit_ktrans = %s\n',num2str(prefs.upper_limit_ktrans));
        fprintf('initial_value_ktrans = %s\n',num2str(prefs.initial_value_ktrans));
        fprintf('lower_limit_fp = %s\n',num2str(prefs.lower_limit_fp));
        fprintf('upper_limit_fp = %s\n',num2str(prefs.upper_limit_fp));
        fprintf('initial_value_fp = %s\n',num2str(prefs.initial_value_fp));
        fprintf('lower_limit_tp = %s\n',num2str(prefs.lower_limit_tp));
        fprintf('upper_limit_tp = %s\n',num2str(prefs.upper_limit_tp));
        fprintf('initial_value_tp = %s\n',num2str(prefs.initial_value_tp));
        fprintf('lower_limit_vp = %s\n',num2str(prefs.lower_limit_vp));
        fprintf('upper_limit_vp = %s\n',num2str(prefs.upper_limit_vp));
        fprintf('initial_value_vp = %s\n',num2str(prefs.initial_value_vp));
    end
    
    if ~USE_GPU
        % Get values from pref file
        prefs_str = parse_preference_file('dce_preferences.txt',0,...
            {'voxel_TolFun' 'voxel_TolX' 'voxel_MaxIter' 'voxel_MaxFunEvals' 'voxel_Robust'});
        prefs.TolFun = str2num(prefs_str.voxel_TolFun);
        prefs.TolX = str2num(prefs_str.voxel_TolX);
        prefs.MaxIter = str2num(prefs_str.voxel_MaxIter);
        prefs.MaxFunEvals = str2num(prefs_str.voxel_MaxFunEvals);
        prefs.Robust = prefs_str.voxel_Robust;
        %Log values used
        if verbose
            fprintf('TolFun = %s\n',num2str(prefs.TolFun));
            fprintf('TolX = %s\n',num2str(prefs.TolX));
            fprintf('MaxIter = %s\n',num2str(prefs.MaxIter));
            fprintf('MaxFunEvals = %s\n',num2str(prefs.MaxFunEvals));
            fprintf('Robust = %s\n',num2str(prefs.Robust));
        end
    
        % Preallocate for speed
        GG = zeros([number_voxels 10],'double');
        residuals = zeros([number_voxels numel(xdata{1}.timer)],'double');
        %Turn off diary if on as it doesn't work with progress bar
        diary_restore = 0;
        if strcmp(get(0,'Diary'),'on')
            diary off;
            diary_restore = 1;
        end
        if verbose; p = ProgressBar(number_voxels,'verbose',verbose); end

        parfor i = 1:number_voxels
            % Do quick linear patlak and use values as initial values
            [estimate, ~] = model_patlak_linear(Ct_data(:,i),Cp_data,timer_data);
            prefs_local = prefs;
            prefs_local.initial_value_ktrans = estimate(1);
            prefs_local.initial_value_vp = estimate(2);
            % Do tissue uptake fit
            [GG(i,:), residuals(i,:)] = model_tissue_uptake(Ct_data(:,i),Cp_data,timer_data,prefs_local);
            if verbose; p.progress; end
        end

        if verbose; p.stop; end
        if diary_restore, diary on, end
    end
    
    if USE_GPU       
        model_id = ModelID.TISSUE_UPTAKE;
        estimator_id = EstimatorID.LSE;
        
        % Load GPU fitting values from pref file
        prefs_str = parse_preference_file('dce_preferences.txt',0,...
            {'gpu_tolerance' 'gpu_max_n_iterations'});
        prefs.gpu_tolerance = str2num(prefs_str.gpu_tolerance);
        prefs.gpu_max_n_iterations = str2num(prefs_str.gpu_max_n_iterations);
        %Log values used
        if verbose
            fprintf('GPU Max iterationos = %s\n',num2str(prefs.gpu_max_n_iterations));
            fprintf('GPU Tolerance = %s\n',num2str(prefs.gpu_tolerance));
        end        
        
        tolerance = prefs.gpu_tolerance;
        max_n_iterations = prefs.gpu_max_n_iterations;
        
        init_param = zeros([3,number_voxels]);
        for i=1:number_voxels
            init_param(1,i) = prefs.initial_value_ktrans;
            init_param(2,i) = prefs.initial_value_vp;
            init_param(3,i) = prefs.initial_value_fp;
        end
        init_param_single = single(init_param);
        
        constraints = zeros([6,number_voxels]);
        for i=1:number_voxels
            constraints(1,i) = prefs.lower_limit_ktrans;
            constraints(2,i) = prefs.upper_limit_ktrans;
            constraints(3,i) = prefs.lower_limit_vp;
            constraints(4,i) = prefs.upper_limit_vp;
            constraints(5,i) = prefs.lower_limit_fp;
            constraints(6,i) = prefs.upper_limit_fp;
        end
        constraints_single = single(constraints);
        % constrain upper and lower bounds for both parameters
        constraint_type = int32([3,3,3]);

        % Load measured data
        indie_vars = single([timer_data' Cp_data]);
        Ct_single = single(Ct_data);
        
        % Execute GPU fit
        [parameters, states, chi_squares, n_iterations, time] = gpufit_constrained(Ct_single,[],model_id,init_param_single,constraints_single,constraint_type,tolerance, max_n_iterations,[],estimator_id,indie_vars);
        %[parameters, states, chi_squares, n_iterations, time] = gpufit(Ct_single,[],model_id,init_param_single,tolerance, max_n_iterations,[],estimator_id,indie_vars);
        state_0 = numel(states(states==0));
        state_1 = numel(states(states==1));
        state_2 = numel(states(states==2));
        state_3 = numel(states(states==3));
        state_4 = numel(states(states==4));
        fprintf('ratio converged = %s\n',num2str(state_0));
        fprintf('ratio max iteration exceeded = %s\n',num2str(state_1));
        fprintf('ratio singular hessian = %s\n',num2str(state_2));
        fprintf('ratio neg curvature MLE = %s\n',num2str(state_3));
        fprintf('ratio gpu not read = %s\n',num2str(state_4));
        % If did not converge discard values
%         one_parameter = parameters(1,:);
%         one_parameter(states~=0) = NaN;  %Ktrans
%         parameters(1,:) = one_parameter;
%         one_parameter = parameters(2,:);
%         one_parameter(states~=0) = NaN;  %vp
%         parameters(2,:) = one_parameter;
%         one_parameter = parameters(3,:);
%         one_parameter(states~=0) = NaN;  %fp
%         parameters(3,:) = one_parameter;
        
        GG = [parameters' chi_squares'];
        % add zeros for the unknown + and - 95 CI
        GG = [GG zeros(number_voxels, 6)];
        residuals = [];
    end
elseif strcmp(model, 'tofts')
    % Slice out needed variables for speed
    Ct_data = xdata{1}.Ct;
    Cp_data = xdata{1}.Cp;
    timer_data = xdata{1}.timer;
    % Get values from pref file
    prefs_str = parse_preference_file('dce_preferences.txt',0,...
        {'voxel_lower_limit_ktrans' 'voxel_upper_limit_ktrans' 'voxel_initial_value_ktrans' ...
        'voxel_lower_limit_ve' 'voxel_upper_limit_ve' 'voxel_initial_value_ve'});
    prefs.lower_limit_ktrans = str2num(prefs_str.voxel_lower_limit_ktrans);
    prefs.upper_limit_ktrans = str2num(prefs_str.voxel_upper_limit_ktrans);
    prefs.initial_value_ktrans = str2num(prefs_str.voxel_initial_value_ktrans);
    prefs.lower_limit_ve = str2num(prefs_str.voxel_lower_limit_ve);
    prefs.upper_limit_ve = str2num(prefs_str.voxel_upper_limit_ve);
    prefs.initial_value_ve = str2num(prefs_str.voxel_initial_value_ve);
    %Log values used
    if verbose
        fprintf('lower_limit_ktrans = %s\n',num2str(prefs.lower_limit_ktrans));
        fprintf('upper_limit_ktrans = %s\n',num2str(prefs.upper_limit_ktrans));
        fprintf('initial_value_ktrans = %s\n',num2str(prefs.initial_value_ktrans));
        fprintf('lower_limit_ve = %s\n',num2str(prefs.lower_limit_ve));
        fprintf('upper_limit_ve = %s\n',num2str(prefs.upper_limit_ve));
        fprintf('initial_value_ve = %s\n',num2str(prefs.initial_value_ve));
    end
    
    if ~USE_GPU
        % Get values from pref file
        prefs_str = parse_preference_file('dce_preferences.txt',0,...
            {'voxel_TolFun' 'voxel_TolX' 'voxel_MaxIter' 'voxel_MaxFunEvals' 'voxel_Robust'});
        prefs.TolFun = str2num(prefs_str.voxel_TolFun);
        prefs.TolX = str2num(prefs_str.voxel_TolX);
        prefs.MaxIter = str2num(prefs_str.voxel_MaxIter);
        prefs.MaxFunEvals = str2num(prefs_str.voxel_MaxFunEvals);
        prefs.Robust = prefs_str.voxel_Robust;
        %Log values used
        if verbose
            fprintf('TolFun = %s\n',num2str(prefs.TolFun));
            fprintf('TolX = %s\n',num2str(prefs.TolX));
            fprintf('MaxIter = %s\n',num2str(prefs.MaxIter));
            fprintf('MaxFunEvals = %s\n',num2str(prefs.MaxFunEvals));
            fprintf('Robust = %s\n',num2str(prefs.Robust));
        end

        % Preallocate for speed
        GG = zeros([number_voxels 7],'double');
        residuals = zeros([number_voxels numel(xdata{1}.timer)],'double');
        %Turn off diary if on as it doesn't work with progress bar
        diary_restore = 0;
        if strcmp(get(0,'Diary'),'on')
            diary off;
            diary_restore = 1;
        end
        if verbose; p = ProgressBar(number_voxels,'verbose',verbose); end
    
        parfor i = 1:number_voxels
            [GG(i,:), residuals(i,:)] = model_tofts(Ct_data(:,i),Cp_data,timer_data,prefs);
            if verbose; p.progress; end
        end
    
        if verbose; p.stop; end
        if diary_restore, diary on, end
    end   
        
    if USE_GPU
        model_id = ModelID.TOFTS;
        estimator_id = EstimatorID.LSE;
        
        % Load GPU fitting values from pref file
        prefs_str = parse_preference_file('dce_preferences.txt',0,...
            {'gpu_tolerance' 'gpu_max_n_iterations'});
        prefs.gpu_tolerance = str2num(prefs_str.gpu_tolerance);
        prefs.gpu_max_n_iterations = str2num(prefs_str.gpu_max_n_iterations);
        %Log values used
        if verbose
            fprintf('GPU Tolerance = %s\n',num2str(prefs.gpu_tolerance));
            fprintf('GPU Max Iterations = %s\n',num2str(prefs.gpu_max_n_iterations));
        end
        
        tolerance = prefs.gpu_tolerance;
        max_n_iterations = prefs.gpu_max_n_iterations;
        
        init_param = zeros([2,number_voxels]);
        for i=1:number_voxels
            init_param(1,i) = prefs.initial_value_ktrans;
            init_param(2,i) = prefs.initial_value_ve;
        end
        init_param_single = single(init_param);
        
        constraints = zeros([4,number_voxels]);
        for i=1:number_voxels
            constraints(1,i) = prefs.lower_limit_ktrans;
            constraints(2,i) = prefs.upper_limit_ktrans; 
            constraints(3,i) = prefs.lower_limit_ve;
            constraints(4,i) = prefs.upper_limit_ve;
        end
        constraints_single = single(constraints);
        % constrain upper and lower bounds for both parameters
        constraint_type = int32([3,3]);

        % Load measured data
        indie_vars = single([timer_data' Cp_data]);
        Ct_single = single(Ct_data);
        
        % Execute GPU fit
        [parameters, states, chi_squares, n_iterations, time] = gpufit_constrained(Ct_single,[],model_id,init_param_single,constraints_single,constraint_type,tolerance, max_n_iterations,[],estimator_id,indie_vars);
        
        % If did not converge discard values
        one_parameter = parameters(1,:);
        one_parameter(states~=0) = NaN;  %Ktrans
        parameters(1,:) = one_parameter;
        one_parameter = parameters(2,:);
        one_parameter(states~=0) = NaN;  %ve
        parameters(2,:) = one_parameter;
        
        GG = [parameters' chi_squares'];
        % add zeros for the unknown + and - 95 CI
        GG = [GG zeros(number_voxels, 4)];
        residuals = [];
    end
elseif strcmp(model, 'fxr')
    % Get values from pref file
    prefs_str = parse_preference_file('dce_preferences.txt',0,...
        {'voxel_lower_limit_ktrans' 'voxel_upper_limit_ktrans' 'voxel_initial_value_ktrans' ...
        'voxel_lower_limit_ve' 'voxel_upper_limit_ve' 'voxel_initial_value_ve' ...
        'voxel_lower_limit_tau' 'voxel_upper_limit_tau' 'voxel_initial_value_tau' ...
        'voxel_TolFun' 'voxel_TolX' 'voxel_MaxIter' 'voxel_MaxFunEvals' 'voxel_Robust'...
        'fxr_fw'});
    prefs.lower_limit_ktrans = str2num(prefs_str.voxel_lower_limit_ktrans);
    prefs.upper_limit_ktrans = str2num(prefs_str.voxel_upper_limit_ktrans);
    prefs.initial_value_ktrans = str2num(prefs_str.voxel_initial_value_ktrans);
    prefs.lower_limit_ve = str2num(prefs_str.voxel_lower_limit_ve);
    prefs.upper_limit_ve = str2num(prefs_str.voxel_upper_limit_ve);
    prefs.initial_value_ve = str2num(prefs_str.voxel_initial_value_ve);
    prefs.lower_limit_tau = str2num(prefs_str.voxel_lower_limit_tau);
    prefs.upper_limit_tau = str2num(prefs_str.voxel_upper_limit_tau);
    prefs.initial_value_tau = str2num(prefs_str.voxel_initial_value_tau);
    prefs.TolFun = str2num(prefs_str.voxel_TolFun);
    prefs.TolX = str2num(prefs_str.voxel_TolX);
    prefs.MaxIter = str2num(prefs_str.voxel_MaxIter);
    prefs.MaxFunEvals = str2num(prefs_str.voxel_MaxFunEvals);
    prefs.Robust = prefs_str.voxel_Robust;
    prefs.fxr_fw = str2num(prefs_str.fxr_fw);
    %Log values used
    if verbose
        fprintf('lower_limit_ktrans = %s\n',num2str(prefs.lower_limit_ktrans));
        fprintf('upper_limit_ktrans = %s\n',num2str(prefs.upper_limit_ktrans));
        fprintf('initial_value_ktrans = %s\n',num2str(prefs.initial_value_ktrans));
        fprintf('lower_limit_ve = %s\n',num2str(prefs.lower_limit_ve));
        fprintf('upper_limit_ve = %s\n',num2str(prefs.upper_limit_ve));
        fprintf('initial_value_ve = %s\n',num2str(prefs.initial_value_ve));
        fprintf('lower_limit_tau = %s\n',num2str(prefs.lower_limit_tau));
        fprintf('upper_limit_tau = %s\n',num2str(prefs.upper_limit_tau));
        fprintf('initial_value_tau = %s\n',num2str(prefs.initial_value_tau));
        fprintf('TolFun = %s\n',num2str(prefs.TolFun));
        fprintf('TolX = %s\n',num2str(prefs.TolX));
        fprintf('MaxIter = %s\n',num2str(prefs.MaxIter));
        fprintf('MaxFunEvals = %s\n',num2str(prefs.MaxFunEvals));
        fprintf('Robust = %s\n',num2str(prefs.Robust));
        fprintf('fxr_fw = %s\n',num2str(prefs.fxr_fw));
    end
    
    % Preallocate for speed
    GG = zeros([number_voxels 10],'double');
    residuals = zeros([number_voxels numel(xdata{1}.timer)],'double');
    % Slice out needed variables for speed
    Ct_data = xdata{1}.Ct;
    Cp_data = xdata{1}.Cp;
    timer_data = xdata{1}.timer;
    R1o= xdata{1}.R1o;
    R1i= xdata{1}.R1i;
    r1 = xdata{1}.relaxivity;
    fw = prefs.fxr_fw;
    %Turn off diary if on as it doesn't work with progress bar
    diary_restore = 0;
    if strcmp(get(0,'Diary'),'on')
        diary off;
        diary_restore = 1;
    end
    if verbose; p = ProgressBar(number_voxels,'verbose',verbose); end;
    parfor i = 1:number_voxels
        [GG(i,:), residuals(i,:)] = model_fxr(Ct_data(:,i),Cp_data,timer_data,R1o(i),R1i(i),r1,fw,prefs);
        if verbose; p.progress; end;
    end;
    if verbose; p.stop; end;
    if diary_restore, diary on, end;
elseif strcmp(model, 'auc')
    % Area under curve using Raw data signal
    
    % Preallocate for speed
    GG = zeros([number_voxels 4],'double');
    % Slice out needed variables for speed
    Sss    = xdata{1}.Sss;
    Ssstum = xdata{1}.Ssstum;
    Stlv   = xdata{1}.Stlv;
    Sttum  = xdata{1}.Sttum;
    Ct_data = xdata{1}.Ct;
    Cp_data = xdata{1}.Cp;
    
    % Substract steady state signal from time curves
    Sttum = Sttum - repmat(Ssstum, [size(Sttum,1) 1]);
    Sss = mean(Sss);
    Stlv  = Stlv - Sss;%repmat(Sss, [size(Stlv,1) 1]);
    %Stlv  = mean(Stlv, 2);
    
    timer_data      = xdata{1}.timer;
    start_injection = xdata{1}.start_injection;
    end_injection   = xdata{1}.end_injection;
    
    % Extract signal only after injection has started
    ind = find(timer_data >= start_injection);
    
    Sttum      = Sttum(ind(1):end,:);
    Stlv       = Stlv(ind(1):end);
    Ct_data    = Ct_data(ind(1):end,:);
    Cp_data    = Cp_data(ind(1):end);
    timer_data = timer_data(ind(1):end);
    
    %Turn off diary if on as it doesn't work with progress bar
    diary_restore = 0;
    if strcmp(get(0,'Diary'),'on')
        diary off;
        diary_restore = 1;
    end
    if verbose; p = ProgressBar(number_voxels,'verbose',verbose); end;
    parfor i = 1:number_voxels
        GG(i,:) = auc_helper(Sttum(:,i),Stlv, Ct_data(:,i), Cp_data, timer_data);
        if verbose; p.progress; end;
    end;
    if verbose; p.stop; end;
    if diary_restore, diary on, end;
    
elseif strcmp(model, 'fractal')
    
elseif strcmp(model, 'nested')
    
    % Get values from pref file
    prefs_str = parse_preference_file('dce_preferences.txt',0,...
        {'voxel_lower_limit_ktrans' 'voxel_upper_limit_ktrans' 'voxel_initial_value_ktrans' ...
        'voxel_lower_limit_ve' 'voxel_upper_limit_ve' 'voxel_initial_value_ve' ...
        'voxel_lower_limit_vp' 'voxel_upper_limit_vp' 'voxel_initial_value_vp' ...
        'voxel_TolFun' 'voxel_TolX' 'voxel_MaxIter' 'voxel_MaxFunEvals' 'voxel_Robust'});
    prefs.lower_limit_ktrans = str2num(prefs_str.voxel_lower_limit_ktrans);
    prefs.upper_limit_ktrans = str2num(prefs_str.voxel_upper_limit_ktrans);
    prefs.initial_value_ktrans = str2num(prefs_str.voxel_initial_value_ktrans);
    prefs.lower_limit_ve = str2num(prefs_str.voxel_lower_limit_ve);
    prefs.upper_limit_ve = str2num(prefs_str.voxel_upper_limit_ve);
    prefs.initial_value_ve = str2num(prefs_str.voxel_initial_value_ve);
    prefs.lower_limit_vp = str2num(prefs_str.voxel_lower_limit_vp);
    prefs.upper_limit_vp = str2num(prefs_str.voxel_upper_limit_vp);
    prefs.initial_value_vp = str2num(prefs_str.voxel_initial_value_vp);
    prefs.TolFun = str2num(prefs_str.voxel_TolFun);
    prefs.TolX = str2num(prefs_str.voxel_TolX);
    prefs.MaxIter = str2num(prefs_str.voxel_MaxIter);
    prefs.MaxFunEvals = str2num(prefs_str.voxel_MaxFunEvals);
    prefs.Robust = prefs_str.voxel_Robust;
    %Log values used
    if verbose
        fprintf('lower_limit_ktrans = %s\n',num2str(prefs.lower_limit_ktrans));
        fprintf('upper_limit_ktrans = %s\n',num2str(prefs.upper_limit_ktrans));
        fprintf('initial_value_ktrans = %s\n',num2str(prefs.initial_value_ktrans));
        fprintf('lower_limit_ve = %s\n',num2str(prefs.lower_limit_ve));
        fprintf('upper_limit_ve = %s\n',num2str(prefs.upper_limit_ve));
        fprintf('initial_value_ve = %s\n',num2str(prefs.initial_value_ve));
        fprintf('lower_limit_vp = %s\n',num2str(prefs.lower_limit_vp));
        fprintf('upper_limit_vp = %s\n',num2str(prefs.upper_limit_vp));
        fprintf('initial_value_vp = %s\n',num2str(prefs.initial_value_vp));
        fprintf('TolFun = %s\n',num2str(prefs.TolFun));
        fprintf('TolX = %s\n',num2str(prefs.TolX));
        fprintf('MaxIter = %s\n',num2str(prefs.MaxIter));
        fprintf('MaxFunEvals = %s\n',num2str(prefs.MaxFunEvals));
        fprintf('Robust = %s\n',num2str(prefs.Robust));
    end
    
    % Preallocate for speed
    GG = zeros([number_voxels 10],'double');
    residuals = zeros([number_voxels numel(xdata{1}.timer)],'double');
    % Slice out needed variables for speed
    Ct_data = xdata{1}.Ct;
    Cp_data = xdata{1}.Cp;
    timer_data = xdata{1}.timer;
    %Turn off diary if on as it doesn't work with progress bar
    diary_restore = 0;
    if strcmp(get(0,'Diary'),'on')
        diary off;
        diary_restore = 1;
    end
    if verbose; p = ProgressBar(number_voxels,'verbose',verbose); end;
    
    parfor i = 1:number_voxels
        % Fit 0 order model
        [GG_zero, residuals(i,:)] = model_0(Ct_data(:,i));
        fp_lower = 0;
        % Fit 1st order model
        [GG_one, residuals_b] = model_vp(Ct_data(:,i),Cp_data,timer_data,prefs);
        fp_higher = 1;
        % Compare
        n = numel(timer_data);
        p_value = ftest(n,fp_lower,fp_higher,GG_zero,GG_one(1,2));
        if p_value>=0.05
            % Use 0 order
            GG_local = zeros([1 10],'double');
            GG_local(1,4) = GG_zero;
            GG(i,:) = GG_local;
            residuals(i,:) = residuals_b';
        else
            % Use 1st order
            GG_local = zeros([1 10],'double');
            GG_local(1,3) = GG_one(1,1);
            GG_local(1,4) = GG_one(1,2);
            GG_local(1,9) = GG_one(1,3);
            GG_local(1,10) = GG_one(1,4);
            GG(i,:) = GG_local;
            residuals(i,:) = residuals_b';
            fp_lower = 1;
            
            % Continue, Fit 2nd order model
            [GG_two, residuals_b] = model_patlak(Ct_data(:,i),Cp_data,timer_data,prefs);
            fp_higher = 2;
            % Compare
            p_value = ftest(n,fp_lower,fp_higher,GG_one(1,2),GG_two(1,3));
            if p_value<0.05
                % Use 2nd order
                GG_local = zeros([1 10],'double');
                GG_local(1,1) = GG_two(1,1);
                GG_local(1,3) = GG_two(1,2);
                GG_local(1,4) = GG_two(1,3);
                GG_local(1,5) = GG_two(1,4);
                GG_local(1,6) = GG_two(1,5);
                GG_local(1,9) = GG_two(1,6);
                GG_local(1,10) = GG_two(1,7);
                GG(i,:) = GG_local;
                residuals(i,:) = residuals_b';
                fp_lower = 2;
                
                % Continue, Fit 3rd order model
                [GG_three, residuals_b] = model_extended_tofts(Ct_data(:,i),Cp_data,timer_data,prefs);
                fp_higher = 3;
                % Compare
                p_value = ftest(n,fp_lower,fp_higher,GG_two(1,3),GG_three(1,4));
                if p_value<0.05
                    % Use 3rd order
                    GG(i,:) = GG_three;
                    residuals(i,:) = residuals_b';
                    fp_lower = 3;
                end
            end
        end
        
        if verbose; p.progress; end
    end
    if verbose; p.stop; end
    if diary_restore, diary on, end
    
elseif strcmp(model, 'patlak')
    % Slice out needed variables for speed
    Ct_data = xdata{1}.Ct;
    Cp_data = xdata{1}.Cp;
    timer_data = xdata{1}.timer;
    % Read preferences
    prefs_str = parse_preference_file('dce_preferences.txt',0,...
        {'voxel_lower_limit_ktrans' 'voxel_upper_limit_ktrans' 'voxel_initial_value_ktrans' ...
        'voxel_lower_limit_vp' 'voxel_upper_limit_vp' 'voxel_initial_value_vp'});
    prefs.lower_limit_ktrans = str2num(prefs_str.voxel_lower_limit_ktrans);
    prefs.upper_limit_ktrans = str2num(prefs_str.voxel_upper_limit_ktrans);
    prefs.initial_value_ktrans = str2num(prefs_str.voxel_initial_value_ktrans);
    prefs.lower_limit_vp = str2num(prefs_str.voxel_lower_limit_vp);
    prefs.upper_limit_vp = str2num(prefs_str.voxel_upper_limit_vp);
    prefs.initial_value_vp = str2num(prefs_str.voxel_initial_value_vp);    
    %Log values used
    if verbose
        fprintf('lower_limit_ktrans = %s\n',num2str(prefs.lower_limit_ktrans));
        fprintf('upper_limit_ktrans = %s\n',num2str(prefs.upper_limit_ktrans));
        fprintf('initial_value_ktrans = %s\n',num2str(prefs.initial_value_ktrans));
        fprintf('lower_limit_vp = %s\n',num2str(prefs.lower_limit_vp));
        fprintf('upper_limit_vp = %s\n',num2str(prefs.upper_limit_vp));
        fprintf('initial_value_vp = %s\n',num2str(prefs.initial_value_vp));
    end
        
    if ~USE_GPU
        % Get CPU fitting values from pref file
        prefs_str = parse_preference_file('dce_preferences.txt',0,...
            {'voxel_TolFun' 'voxel_TolX' 'voxel_MaxIter' 'voxel_MaxFunEvals' 'voxel_Robust'});
        prefs.TolFun = str2num(prefs_str.voxel_TolFun);
        prefs.TolX = str2num(prefs_str.voxel_TolX);
        prefs.MaxIter = str2num(prefs_str.voxel_MaxIter);
        prefs.MaxFunEvals = str2num(prefs_str.voxel_MaxFunEvals);
        prefs.Robust = prefs_str.voxel_Robust;
        %Log values used
        if verbose
            fprintf('TolFun = %s\n',num2str(prefs.TolFun));
            fprintf('TolX = %s\n',num2str(prefs.TolX));
            fprintf('MaxIter = %s\n',num2str(prefs.MaxIter));
            fprintf('MaxFunEvals = %s\n',num2str(prefs.MaxFunEvals));
            fprintf('Robust = %s\n',num2str(prefs.Robust));
        end
    
        % Preallocate for speed
        GG = zeros([number_voxels 7],'double');
        residuals = zeros([number_voxels numel(xdata{1}.timer)],'double');
        %Turn off diary if on as it doesn't work with progress bar
        diary_restore = 0;
        if strcmp(get(0,'Diary'),'on')
            diary off;
            diary_restore = 1;
        end
        if verbose; p = ProgressBar(number_voxels,'verbose',verbose); end
        
        parfor i = 1:number_voxels
            % Do quick linear patlak and use values as initial values
            [estimate, ~] = model_patlak_linear(Ct_data(:,i),Cp_data,timer_data);
            prefs_local = prefs;
            prefs_local.initial_value_ktrans = estimate(1);
            prefs_local.initial_value_vp = estimate(2);
            % Do non-linear patlak
            [GG(i,:), residuals(i,:)] = model_patlak(Ct_data(:,i),Cp_data,timer_data,prefs_local);
            if verbose; p.progress; end;
        end
    
        if verbose; p.stop; end
        if diary_restore, diary on, end
    else
        model_id = ModelID.PATLAK;
        estimator_id = EstimatorID.LSE;
        
        % Load GPU fitting values from pref file
        prefs_str = parse_preference_file('dce_preferences.txt',0,...
            {'gpu_tolerance' 'gpu_max_n_iterations'});
        prefs.gpu_tolerance = str2num(prefs_str.gpu_tolerance);
        prefs.gpu_max_n_iterations = str2num(prefs_str.gpu_max_n_iterations);
        %Log values used
        if verbose
            fprintf('GPU Tolerance = %s\n',num2str(prefs.gpu_tolerance));
            fprintf('GPU Max Iterations = %s\n',num2str(prefs.gpu_max_n_iterations));
        end
        
        tolerance = prefs.gpu_tolerance;
        max_n_iterations = prefs.gpu_max_n_iterations;

        init_param = zeros([2,number_voxels]);
        for i=1:number_voxels
            init_param(1,i) = prefs.initial_value_ktrans;
            init_param(2,i) = prefs.initial_value_vp;
        end
        init_param_single = single(init_param);
        
        constraints = zeros([4,number_voxels]);
        for i=1:number_voxels
            constraints(1,i) = prefs.lower_limit_ktrans;
            constraints(2,i) = prefs.upper_limit_ktrans;
            constraints(3,i) = prefs.lower_limit_vp;
            constraints(4,i) = prefs.upper_limit_vp;
        end
        constraints_single = single(constraints);
        % constrain upper and lower bounds for both parameters
        constraint_type = int32([3,3]);
        
        % Load measured data
        % Ensure timer_data is a row vector and not the same size as Cp_data
        if size(timer_data, 1) == size(Cp_data, 1) && size(timer_data, 2) == size(Cp_data, 2)
            timer_data = timer_data';
        end
        if size(timer_data, 1) > 1 && size(timer_data, 2) == 1
            timer_data = timer_data';
        end
        indie_vars = single([timer_data Cp_data]);
        Ct_single = single(Ct_data);
        weights = [];
        % start_i = input("Start:");
        % end_i = input("End:");
        weights = single(ones(size(Ct_single)));
        % weights(start_i:end_i, :) = 0;
        weights = weights(1:size(Ct_single, 1), 1:size(Ct_single,2));
        % Read weights from ODS file
        % [~, ~, raw] = xlsread('/media/network_mriphysics/USC-animal/BIDS/code/brain_avg_scores.xlsx');
        % ids = raw(2:end,1); % skip header
        % weight_starts = cell2mat(raw(2:end,7));
        % weight_ends = cell2mat(raw(2:end,8));

        % Find matching ID for current subject/session
        if isfield(xdata{1}, 'ID')
            current_id = xdata{1}.ID;
        else
            error('xdata{1} must contain an ''ID'' field matching the ODS file first column.');
        end

        row_idx = find(strcmp(ids, current_id), 1);
        if isempty(row_idx)
            error(['ID ' current_id ' not found in ODS file.']);
        end

        % Handle NaN or empty cells for start/end indices
        %TODO - automate with start_t choppage
        % start_i = max(1, weight_starts(row_idx)-10);
        % end_i = max(1, weight_ends(row_idx)-10);

        if isempty(start_i) || isnan(start_i) || isempty(end_i) || isnan(end_i)
            % If missing, do not mask any weights
            % warning(['Missing or NaN weight range for ID ' current_id '. No weights will be masked.']);
            weights = single(ones(size(Ct_single)));
        else
            % Set weights to zero in the specified range
            weights = single(ones(size(Ct_single)));
            weights(start_i:end_i, :) = 0;
            weights = weights(1:size(Ct_single, 1), 1:size(Ct_single,2));
        end
        
        % Execute GPU fit
        [parameters, states, chi_squares, n_iterations, time] = gpufit_constrained(Ct_single,weights,model_id,init_param_single,constraints_single, constraint_types, tolerance, max_n_iterations,[],estimator_id,indie_vars);

        % If did not converge discard values
        one_parameter = parameters(1,:);
        one_parameter(states~=0) = NaN;  %Ktrans
        parameters(1,:) = one_parameter;
        one_parameter = parameters(2,:);
        one_parameter(states~=0) = NaN;  %vp
        parameters(2,:) = one_parameter;
        
        GG = [parameters' chi_squares'];
        % add zeros for the unknown + and - 95 CI
        GG = [GG zeros(number_voxels, 4)];
        residuals = [];
        
%         for i=1:number_voxels
%             % filter negatives
%             if parameters(1,i) > 0
%                 GG(i,1) = parameters(1,i); %Ktrans
%             end
%             GG(i,2) = parameters(2,i); %vp
%         end
    end   
   
elseif strcmp(model, 'patlak_linear')

    % Preallocate for speed
    GG = zeros([number_voxels 7],'double');
    residuals = zeros([number_voxels numel(xdata{1}.timer)],'double');
    % Slice out needed variables for speed
    Ct_data = xdata{1}.Ct;
    Cp_data = xdata{1}.Cp;
    timer_data = xdata{1}.timer;
    %Turn off diary if on as it doesn't work with progress bar
    diary_restore = 0;
    if strcmp(get(0,'Diary'),'on')
        diary off;
        diary_restore = 1;
    end
    if verbose; p = ProgressBar(number_voxels,'verbose',verbose); end
    parfor i = 1:number_voxels
        [GG(i,:), residuals(i,:)] = model_patlak_linear(Ct_data(:,i),Cp_data,timer_data);
        if verbose; p.progress; end
    end
    if verbose; p.stop; end
    if diary_restore, diary on, end
    
elseif strcmp(model, '2cxm')
    % Slice out needed variables for speed
    Ct_data = xdata{1}.Ct;
    Cp_data = xdata{1}.Cp;
    timer_data = xdata{1}.timer;
    % Get values from pref file
    prefs_str = parse_preference_file('dce_preferences.txt',0,...
        {'voxel_lower_limit_ktrans' 'voxel_upper_limit_ktrans' 'voxel_initial_value_ktrans' ...
        'voxel_lower_limit_ve' 'voxel_upper_limit_ve' 'voxel_initial_value_ve' ...
        'voxel_lower_limit_vp' 'voxel_upper_limit_vp' 'voxel_initial_value_vp' ...
        'voxel_lower_limit_fp' 'voxel_upper_limit_fp' 'voxel_initial_value_fp'});
    prefs.lower_limit_ktrans = str2num(prefs_str.voxel_lower_limit_ktrans);
    prefs.upper_limit_ktrans = str2num(prefs_str.voxel_upper_limit_ktrans);
    prefs.initial_value_ktrans = str2num(prefs_str.voxel_initial_value_ktrans);
    prefs.lower_limit_ve = str2num(prefs_str.voxel_lower_limit_ve);
    prefs.upper_limit_ve = str2num(prefs_str.voxel_upper_limit_ve);
    prefs.initial_value_ve = str2num(prefs_str.voxel_initial_value_ve);
    prefs.lower_limit_vp = str2num(prefs_str.voxel_lower_limit_vp);
    prefs.upper_limit_vp = str2num(prefs_str.voxel_upper_limit_vp);
    prefs.initial_value_vp = str2num(prefs_str.voxel_initial_value_vp);
    prefs.lower_limit_fp = str2num(prefs_str.voxel_lower_limit_fp);
    prefs.upper_limit_fp = str2num(prefs_str.voxel_upper_limit_fp);
    prefs.initial_value_fp = str2num(prefs_str.voxel_initial_value_fp);
    %Log values used
    if verbose
        fprintf('lower_limit_ktrans = %s\n',num2str(prefs.lower_limit_ktrans));
        fprintf('upper_limit_ktrans = %s\n',num2str(prefs.upper_limit_ktrans));
        fprintf('initial_value_ktrans = %s\n',num2str(prefs.initial_value_ktrans));
        fprintf('lower_limit_ve = %s\n',num2str(prefs.lower_limit_ve));
        fprintf('upper_limit_ve = %s\n',num2str(prefs.upper_limit_ve));
        fprintf('initial_value_ve = %s\n',num2str(prefs.initial_value_ve));
        fprintf('lower_limit_vp = %s\n',num2str(prefs.lower_limit_vp));
        fprintf('upper_limit_vp = %s\n',num2str(prefs.upper_limit_vp));
        fprintf('initial_value_vp = %s\n',num2str(prefs.initial_value_vp));
        fprintf('lower_limit_fp = %s\n',num2str(prefs.lower_limit_fp));
        fprintf('upper_limit_fp = %s\n',num2str(prefs.upper_limit_fp));
        fprintf('initial_value_fp = %s\n',num2str(prefs.initial_value_fp));
    end
    
    if ~USE_GPU
        % Get values from pref file
        prefs_str = parse_preference_file('dce_preferences.txt',0,...
            {'voxel_TolFun' 'voxel_TolX' 'voxel_MaxIter' 'voxel_MaxFunEvals' 'voxel_Robust'});
        prefs.TolFun = str2num(prefs_str.voxel_TolFun);
        prefs.TolX = str2num(prefs_str.voxel_TolX);
        prefs.MaxIter = str2num(prefs_str.voxel_MaxIter);
        prefs.MaxFunEvals = str2num(prefs_str.voxel_MaxFunEvals);
        prefs.Robust = prefs_str.voxel_Robust;
        %Log values used
        if verbose
            fprintf('TolFun = %s\n',num2str(prefs.TolFun));
            fprintf('TolX = %s\n',num2str(prefs.TolX));
            fprintf('MaxIter = %s\n',num2str(prefs.MaxIter));
            fprintf('MaxFunEvals = %s\n',num2str(prefs.MaxFunEvals));
            fprintf('Robust = %s\n',num2str(prefs.Robust));
        end
    
        % Preallocate for speed
        GG = zeros([number_voxels 13],'double');
        residuals = zeros([number_voxels numel(xdata{1}.timer)],'double');
        %Turn off diary if on as it doesn't work with progress bar
        diary_restore = 0;
        if strcmp(get(0,'Diary'),'on')
            diary off;
            diary_restore = 1;
        end
        if verbose; p = ProgressBar(number_voxels,'verbose',verbose); end
        
        parfor i = 1:number_voxels
            [GG(i,:), residuals(i,:)] = model_2cxm(Ct_data(:,i),Cp_data,timer_data,prefs);
            if verbose; p.progress; end
        end
        if verbose; p.stop; end
        if diary_restore, diary on, end
    end
    
    if USE_GPU
        model_id = ModelID.TWO_COMPARTMENT_EXCHANGE;
        estimator_id = EstimatorID.LSE;
        
        % Load GPU fitting values from pref file
        prefs_str = parse_preference_file('dce_preferences.txt',0,...
            {'gpu_tolerance' 'gpu_max_n_iterations'});
        prefs.gpu_tolerance = str2num(prefs_str.gpu_tolerance);
        prefs.gpu_max_n_iterations = str2num(prefs_str.gpu_max_n_iterations);
        %Log values used
        if verbose
            fprintf('GPU Tolerance = %s\n',num2str(prefs.gpu_tolerance));
            fprintf('GPU Max Iterations = %s\n',num2str(prefs.gpu_max_n_iterations));
        end
        
        tolerance = prefs.gpu_tolerance;
        max_n_iterations = prefs.gpu_max_n_iterations;
        
        init_param = zeros([4,number_voxels]);
        for i=1:number_voxels
            init_param(1,i) = prefs.initial_value_ktrans;
            init_param(2,i) = prefs.initial_value_ve;
            init_param(3,i) = prefs.initial_value_vp;
            init_param(4,i) = prefs.initial_value_fp;
        end
        init_param_single = single(init_param);
        
        constraints = zeros([8,number_voxels]);
        for i=1:number_voxels
            constraints(1,i) = prefs.lower_limit_ktrans;
            constraints(2,i) = prefs.upper_limit_ktrans;
            constraints(3,i) = prefs.lower_limit_ve;
            constraints(4,i) = prefs.upper_limit_ve;
            constraints(5,i) = prefs.lower_limit_vp;
            constraints(6,i) = prefs.upper_limit_vp;
            constraints(7,i) = prefs.lower_limit_fp;
            constraints(8,i) = prefs.upper_limit_fp;
        end
        constraints_single = single(constraints);
        % constrain upper and lower bounds for both parameters
        constraint_type = int32([3,3,3,3]);

        % Load measured data
        indie_vars = single([timer_data' Cp_data]);
        Ct_single = single(Ct_data);
        
        % Execute GPU fit
        [parameters, states, chi_squares, n_iterations, time] = gpufit_constrained(Ct_single,[],model_id,init_param_single,constraints_single,constraint_type,tolerance, max_n_iterations,[],estimator_id,indie_vars);
        
        % If did not converge discard values
        one_parameter = parameters(1,:);
        one_parameter(states~=0) = NaN;  %Ktrans
        parameters(1,:) = one_parameter;
        one_parameter = parameters(2,:);
        one_parameter(states~=0) = NaN;  %ve
        parameters(2,:) = one_parameter;
        one_parameter = parameters(3,:);
        one_parameter(states~=0) = NaN;  %vp
        parameters(3,:) = one_parameter;
        one_parameter = parameters(4,:);
        one_parameter(states~=0) = NaN;  %fp
        parameters(4,:) = one_parameter;
        
        GG = [parameters' chi_squares'];
        % add zeros for the unknown + and - 95 CI
        GG = [GG zeros(number_voxels, 8)];
        residuals = [];
    end
    
elseif strcmp(model, 'FXL_rr')    
    % Get values from pref file
    prefs_str = parse_preference_file('dce_preferences.txt',0,...
        {'voxel_lower_limit_ktrans' 'voxel_upper_limit_ktrans' 'voxel_initial_value_ktrans' ...
        'voxel_lower_limit_ve' 'voxel_upper_limit_ve' 'voxel_initial_value_ve' ...
        'voxel_lower_limit_ktrans_RR' 'voxel_upper_limit_ktrans_RR' 'voxel_initial_value_ktrans_RR' ...
        'voxel_value_ve_RR' ...
        'voxel_TolFun' 'voxel_TolX' 'voxel_MaxIter' 'voxel_MaxFunEvals' 'voxel_Robust'});
    prefs.lower_limit_ktrans = str2num(prefs_str.voxel_lower_limit_ktrans);
    prefs.upper_limit_ktrans = str2num(prefs_str.voxel_upper_limit_ktrans);
    prefs.initial_value_ktrans = str2num(prefs_str.voxel_initial_value_ktrans);
    prefs.lower_limit_ve = str2num(prefs_str.voxel_lower_limit_ve);
    prefs.upper_limit_ve = str2num(prefs_str.voxel_upper_limit_ve);
    prefs.initial_value_ve = str2num(prefs_str.voxel_initial_value_ve);
    prefs.lower_limit_ktrans_RR = str2num(prefs_str.voxel_lower_limit_ktrans_RR);
    prefs.upper_limit_ktrans_RR = str2num(prefs_str.voxel_upper_limit_ktrans_RR);
    prefs.initial_value_ktrans_RR = str2num(prefs_str.voxel_initial_value_ktrans_RR);
    prefs.value_ve_RR = str2num(prefs_str.voxel_value_ve_RR);
    prefs.TolFun = str2num(prefs_str.voxel_TolFun);
    prefs.TolX = str2num(prefs_str.voxel_TolX);
    prefs.MaxIter = str2num(prefs_str.voxel_MaxIter);
    prefs.MaxFunEvals = str2num(prefs_str.voxel_MaxFunEvals);
    prefs.Robust = prefs_str.voxel_Robust;
    %Log values used
    if verbose
        fprintf('lower_limit_ktrans = %s\n',num2str(prefs.lower_limit_ktrans));
        fprintf('upper_limit_ktrans = %s\n',num2str(prefs.upper_limit_ktrans));
        fprintf('initial_value_ktrans = %s\n',num2str(prefs.initial_value_ktrans));
        fprintf('lower_limit_ve = %s\n',num2str(prefs.lower_limit_ve));
        fprintf('upper_limit_ve = %s\n',num2str(prefs.upper_limit_ve));
        fprintf('initial_value_ve = %s\n',num2str(prefs.initial_value_ve));
        fprintf('lower_limit_ktrans_RR = %s\n',num2str(prefs.lower_limit_ktrans_RR));
        fprintf('upper_limit_ktrans_RR = %s\n',num2str(prefs.upper_limit_ktrans_RR));
        fprintf('initial_value_ktrans_RR = %s\n',num2str(prefs.initial_value_ktrans_RR));
        fprintf('value_ve_RR = %s\n',num2str(prefs.value_ve_RR));
        fprintf('TolFun = %s\n',num2str(prefs.TolFun));
        fprintf('TolX = %s\n',num2str(prefs.TolX));
        fprintf('MaxIter = %s\n',num2str(prefs.MaxIter));
        fprintf('MaxFunEvals = %s\n',num2str(prefs.MaxFunEvals));
        fprintf('Robust = %s\n',num2str(prefs.Robust));
    end
    
    % Preallocate for speed
    GG = zeros([number_voxels 10],'double');
    residuals = zeros([number_voxels numel(xdata{1}.timer)],'double');
    % Slice out needed variables for speed
    Ct_data = xdata{1}.Ct;
    Cp_data = xdata{1}.Cp;
    timer_data = xdata{1}.timer;
    %Turn off diary if on as it doesn't work with progress bar
    diary_restore = 0;
    if strcmp(get(0,'Diary'),'on')
        diary off;
        diary_restore = 1;
    end
    if verbose; p = ProgressBar(number_voxels,'verbose',verbose); end;
    parfor i = 1:number_voxels 

        [GG(i,:), residuals(i,:)] = model_FXL_reference_region(Ct_data(:,i),Cp_data,timer_data,prefs);
        if verbose; p.progress; end
    end
    if verbose; p.stop; end
    if diary_restore, diary on, end
    
else
    error(['Error, model ' model ' not yet implemented']);
end
