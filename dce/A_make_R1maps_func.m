% function saved_results = A_make_R1maps_func(DYNAMIC, LV, TUMOR, NOISE, DRIFT, hdr, res,quant, rootname, dynampath, dynam_name, aif_rr_type, ... 
function [saved_results, RUNA_vars, errormsg] = A_make_R1maps_func(filevolume, noise_pathpick, ...
    noise_pixsize, LUT, filelist, t1aiffiles, t1roifiles, t1mapfiles, noisefiles, ...
    driftfiles, rootname, fileorder, quant, mask_roi, mask_aif, ...
    aif_rr_type, tr, fa, hematocrit, snr_filter, relaxivity, ...
    steady_state_time, drift_global, blood_t1, injection_duration, start_t, end_t, save_output)

% A_make_R1maps_func - Generate concentration versus time curves for the
% tumor region and the arterial input region. The setup follows Loveless
% et. al 2011 MRM method of AIF filtering based upon SNR.
% 
% Inputs:
%  DYNAMIC            - Image matrix that contains the dynamic
%                       dataset. Formated as volume x time points.
%  LV                 - Image matrix  that contains the T1 map of
%                       the arterial input region (or the reference region
%                       in the case of the reference region method). All
%                       values outside of the AIF ROI set to <=0
%  TUMOR              - Image matrix  that contains the T1 map of the DCE
%                       region, the region that will have DCE values
%                       calculated. All values outside of the ROI set to
%                       <=0
%  NOISE              - Image matrix  that delineates a noise region of the
%                       image (not T1 map). Used for SNR calculation. All
%                       values outside of the ROI set to <=0
%  DRIFT              - Image matrix  that delineates a region of the image
%                       (not T1 map) for drift correction (e.g. water 
%                       phantom). All values outside of the ROI set to <=0
%  hdr                - Image header of NIFTI or DICOM file read from DCE
%                       region
%  res                - vector with voxel size
%  quant              - boolean value indicating whether to pursue
%                       quantitative DCE vs. semi-quant values
%  rootname           - Name of output file
%  dynampath          - Name of output path
%  dynamname          - Name of final file for part D
%  aif_rr_type        - Indicates the type of auxiliary ROI, as reference 
%                       region 'rr' or arterial input 'aif_roi'. May also
%                       indicate if the AIF is to be auto found 'aif_auto'
%                       or auto found with a static T1 value
%                       'aif_auto_static'
%  tr                 - reptition time (in ms) of dynamic scan
%  fa                 - flip angle (in degrees) of dynamic scan
%  hematocrit         - hematocrit percent (0 - 1.00) of subject
%  snr_filter         - snr required for AIF voxels, snr must exceed this
%                       value averaged over all time points
%  relaxivity         - r1 relaxivity (in mM^-1*sec^-1) of contrast agent
%  steady_state_time  - in image number, defines the end of the steady
%                       state period before contrast injection. -1
%                       indicates users will be prompted to select it
%                       graphically
%  drift_global       - boolean value, perform drift correction globally,
%                       not on a slice by slice basis
%  blood_t1           - the T1 value of blood (in ms) only used if 
%                       aif_rr_type is set to 'aif_auto_static'
%  injection_duration - the length of the contrast agent injection in
%                       number of acquisitions, for auto AIF only
% 
% We assume that the T1 maps and the DCE-MRI files contain the same field of
% view.
% 
% The script uses the T1 maps to calculate the R1 changes over time for the
% dynamic dataset using the gradient echo signal equation. 
% 
% Voxels whereby the R1 calculation creates unreasonable values (e.g. complex
% values) are filtered out.
% 
% For the AIF (or reference region), a SNR filter is applied. Voxels that
% have SNR less than denoted will be filtered out. 
% 
% Concentration maps and time curves are saved. 
% 
% A matlab data .mat file is saved for further processing.
% 
% Requires:
% A_make_R1maps_func.m
% DataHash.m
% cleanAB.m
% cleanR1t.m
% findRod.m
% niftitools
% 
% Thomas Ng
% Caltech, Dec 2011
% Updated April 2012
% Updated Dec 2013 Sam Barnes
% Updated Jan 2014 Thomas Ng




%% 1. Option Toggles

% Toggle to re-calculate the average values only in the regions that the
% voxel curvefit was able to give good fitting; ie. the viable regions.
viable= 0;
if ispc
    opengl('software');
else
    disp('Non PC unable to run OpenGL software mode, there may be some graphics issues')
end
if nargin < 28
    save_output = true;
end


%% DO NOT ALTER LINES BELOW UNLESS YOU KNOW WHAT YOU ARE DOING
% Log input results
%[log_path,base,~] = fileparts(dce_path);
[dynampath, ~]  = fileparts(filelist{1});
log_path = dynampath;
log_path = fullfile(log_path, ['A_' rootname 'R1info.log']);
if exist(log_path, 'file')==2
  delete(log_path);
end
diary(log_path);

fprintf('************** User Input **************\n\n');
% disp('User selected dce file: ');
% fprintf('%s\n\n',dce_path);
% disp('User selected T1 of AIF: ');
% fprintf('%s\n\n',t1_aif_path);
% disp('User selected T1 of region to process: ');
% fprintf('%s\n\n',t1_roi_path);
% disp('User selected noise region in image (not T1 map): ');
% fprintf('%s\n\n',noise_path);
disp('User selected TR (ms): ');
disp(tr);
disp('User selected FA (degrees): ');
disp(fa);
disp('User selected hematocit (0 to 1.0): ');
disp(hematocrit);
disp('User selected SNR threshold for AIF: ');
disp(snr_filter);
disp('User selected contrast agent R1 relaxivity (/mM/sec): ');
disp(relaxivity);
disp('User selected end of steady state time (image number): ');
disp(steady_state_time);

if quant
    disp('User selected quantification: ie. T1 maps');
else
    disp('User selected no quantification: raw signal data only');
end


disp('User selected starting repetition: ')
if ~isempty(start_t)
    disp(start_t)
else
    disp('None')
end

disp('User selected ending repetition: ')
if ~isempty(end_t)
    disp(end_t)
else
    disp('None')
end

%% 2. a) Load the files

[TUMOR, LV, NOISE, DYNAMIC, DRIFT, dynampath, dynam_name, rootname, hdr, ...
    res, sliceloc, errormsg] = ...
    loadIMGVOL(filevolume, noise_pathpick, noise_pixsize, LUT, t1aiffiles, ...
    t1roifiles, t1mapfiles, noisefiles, driftfiles, filelist, rootname, ...
    fileorder, mask_roi, mask_aif, start_t, end_t);

disp('User selected drift correction: ');
if(isempty(find(DRIFT > 0, 1)))
    disp('None')
else
    if(drift_global)
        disp('Global');
    else
        disp('Slicewise');
    end
end
fprintf('************** End User Input **************\n\n\n');
disp('Starting Part A Processing')
disp(datetime)
disp(' ');
tic

% Ask for file location
place = '';
[PathName1,~,~] = fileparts(log_path);

% [PathName1,base,ext] = fileparts(dce_path);
% dynam = [base ext];
% 
% [PathName2,base,ext] = fileparts(t1_aif_path);
% lv = [base ext];
% 
% [PathName3,base,ext] = fileparts(t1_roi_path);
% tumor = [base ext];
% 
% [PathName4,base,ext] = fileparts(noise_path);
% noise = [base ext];
% 
% % Output name
% rootname = strrep(dynam,'.nii','_');

% %Load DCE dataset
% dynam = load_untouch_nii(fullfile(PathName1, place, dynam));
% 
% %image resolution
% res  = dynam.hdr.dime.pixdim;
% save for part D
%dynam.fileprefix;

dynam = DYNAMIC; %double(dynam.img);


%Load TUMOR T1 map, find the voxels that encompass tumor ROI
%convert T1 from ms to sec
% TUMOR = load_untouch_nii(fullfile(PathName3, place, tumor));
% TUMOR = 1/1000.*double(TUMOR.img);
tumind= find(TUMOR > 0);

testt1 = mean(TUMOR(tumind));
if quant
    if testt1 > 50
        disp('T1 maps of data likely in ms, converting to s...');
        TUMOR = (1/1000).*double(TUMOR);
    else
        disp('T1 maps of data likely in sec, no conversion');
    end
end



%Load AIF dataset, convert T1 from ms to sec
lvind = find(LV > 0);
%If static blood T1, insert static value now
if strcmp(aif_rr_type,'aif_roi_static')
    LV(lvind) = blood_t1;
end
%Check units convert to sec if required
testt1 = mean(LV(lvind));
if quant
    if testt1 > 50
        disp('T1 maps of AIF likely in ms, converting to s...');
        LV = (1/1000).*double(LV);
    else
        disp('T1 maps of AIF likely in sec, no conversion');
    end
end

%Load noise ROI files
% NOISE = load_untouch_nii(fullfile(PathName4, place, noise));
% NOISE = double(NOISE.img);
noiseind = find(NOISE > 0);

%Load drift ROI files
driftind = find(DRIFT > 0);

% If viable toggle is on, choose the file that is generated with the
% viable tumor region.
if(viable)
    [VIA,PathName1,FilterIndex] = uigetfile([PathName1 '/*.nii'],'Choose Ktrans file Region file');
    VIA = load_untouch_nii(fullfile(PathName1, place, VIA));
    VIA = VIA.img;
    viaind = find(VIA ~=0);
    nonvia = setdiff(tumind, viaind);
    disp([num2str(100*numel(viaind)/numel(tumind)) '% of the tumor gives good fitting and is kept for further analysis'])
    
    tumind = viaind;
end

%Find the number of slices each volume dataset
% 
[dimx, dimy, dimz, dimt] = size(dynam);
if dimt==1
    % Check if time is the third dimension
    dimz_check = size(TUMOR, 3);
    if dimz_check==1 && dimz~=1
        %if so reshpae to be 4D matrix
        dimt = dimz;
        dimz = 1;
        dynam = reshape(dynam,dimx,dimy,dimz,dimt);
    else
        error('Input dynamic images not 4D data, or time dimension = 1');
    end
    
end

% The TR is in ms
tr = tr/1000;

%% 3. Setup the data for next step.

warning off
% Initialize the time curve arrays
DYNAM       = [];
DYNAMLV     = [];
DYNAMNOISE  = [];
DYNAMNONVIA = [];

if dimz>10
    mod_dimz = ceil(dimz/10);
    plot_dimz = floor(dimz/mod_dimz);
else
    plot_dimz = dimz;
    mod_dimz = 1;
end

for i = 1:dimt
    %For each time point, we collect the dynamic information into the curve
    %arrays.
    
    currentimg       = dynam(:,:,:,i);
    DYNAM(end+1,:)   = currentimg(tumind);
    DYNAMLV(end+1,:) = currentimg(lvind);
    DYNAMNOISE(end+1)= std(single(currentimg(noiseind)));
    
    if(i == 1 && exist('prctile', 'file'))
        % This is used for create a graphic showing the ROIs in relation to the
        % DCE MRI image.
        matchimg = currentimg;
        size_image_2d = [size(matchimg,1),size(matchimg,2)];
        size_image_3d = [size(matchimg,1),size(matchimg,2), dimz];

%         red_mask = cat(3, ones(size_image_2d)', zeros(size_image_2d)', zeros(size_image_2d)');
%         green_mask = cat(3, zeros(size_image_2d)', ones(size_image_2d)', zeros(size_image_2d)');
%         blue_mask = cat(3, zeros(size_image_2d)', zeros(size_image_2d)', ones(size_image_2d)');
%         yellow_mask = cat(3, zeros(size_image_2d)', ones(size_image_2d)', ones(size_image_2d)');
%         cyan_mask = cat(3, zeros(size_image_2d)', ones(size_image_2d)', ones(size_image_2d)');
        region_mask = zeros(size_image_3d);
        aif_mask = zeros(size_image_3d);
        noise_mask = zeros(size_image_3d);
        drift_mask = zeros(size_image_3d);
        nonviable_mask = zeros(size_image_3d);
        region_mask(tumind) = 1;
        noise_mask(noiseind)=1;
        if ~(strcmp(aif_rr_type,'aif_auto') || strcmp(aif_rr_type,'aif_auto_static'))
            aif_mask(lvind)  = 1;
        end
        drift_mask(driftind) = 1;
        if(viable)
            nonviable_mask(nonvia)=1;
        end

        nn = figure;
        % Permute x and y
        imshow3D_overlays(permute(matchimg,[2 1 3]),...
            [prctile(reshape(matchimg,1,[]),5) prctile(reshape(matchimg,1,[]),95)],...
            permute(region_mask,[2 1 3]),...
            permute(noise_mask,[2 1 3]),...
            permute(aif_mask,[2 1 3]),...
            permute(nonviable_mask,[2 1 3]),...
            permute(drift_mask,[2 1 3])...
            );
        drawnow
    else
        warning('ROIs relative to DCE image not drawn due to lack of function "prctile". Your version of MATLAB may require the Statistics and Machine Learning Toolbox')
    end
    
end

%% 4. Correcting drift

% If for some reason the MR signal drifts over time, and you have included
% a rod phantom in the FOV during the imaging, you can use the drift of the
% phantom to correct for the MR signal in the tissue of interest.

if(~isempty(driftind))   
    % Drift correct the image
    if(drift_global)
        %do global correction
        scale_raw = ones(1,dimt);
        
        % First get time curve of rod and fit to poly
        %Gets 3d image at the second time point, this is the baseline
        %signal
        baseline_img = dynam(:,:,:,2);
        
        % Time loop
        for t = 1:dimt
            current_img = dynam(:,:,:,t);
            % calculate signal relative to that at the baseline singal(2nd
            % timepoint), this is scale_raw
            scale_raw(t) = mean(baseline_img(driftind))/mean(current_img(driftind));
        end
        
        % Fit time curve of rod to poly4
        scale_fit = fit((1:numel(scale_raw))',scale_raw','poly4','Robust','on');
        
        % Second Scale images
        DYNAM       = [];
        DYNAMLV     = [];
        DYNAMNOISE  = [];
        DYNAMNONVIA = [];

        % Time loop
        for time_index = 1:dimt
            % All slices at the time time_index
            currentimg       = dynam(:,:,:,time_index);
            

            DRIFT_SIGNAL(time_index) = mean(currentimg(driftind));
            PRECORRECTED(time_index) = mean(currentimg(:));
            currentimg = currentimg.*scale_fit(time_index);
            CORRECTED(time_index) = mean(currentimg(:));


            DYNAM(end+1,:)   = currentimg(tumind);
            DYNAMLV(end+1,:) = currentimg(lvind);
            DYNAMNOISE(end+1)= std(single(currentimg(noiseind)));

            if(viable)
                DYNAMNONVIA(end+1,:) = currentimg(nonvia);
            end
        end
        
        % Plot the drift and signal pre/post correction
        drift_fig = figure;

        subplot(2, 1, 1)
        hold on
            plot(DRIFT_SIGNAL(:)./(DRIFT_SIGNAL(2)*.01), 'r.')
            plot(100./scale_fit(1:time_index),'k')
        hold off
        title('Drift phantom signal and fit');
        legend('phantom','poly fit')
        subplot(2,1,2)
        hold on
            plot(PRECORRECTED(:), 'b.')
            plot(CORRECTED(:), 'gx')
        hold off
        title('Drift Corrected and Uncorrected Mean Signal');
        legend('uncorrected','corrected', 'Location', 'southeast')

        saveas(drift_fig, fullfile(PathName1, [rootname '_drift.fig']));
            
    else
        %do slice by slice correction
    
        % First get time curve of rod and fit to poly
        %Gets 3d image at the second time point, this is the baseline
        %signal
        baseline_img = dynam(:,:,:,2);
        scale_fit = cell(1,dimz);
        % Slice loop
        for j = 1:dimz 
            % Slice j from baseline
            originalimgj = baseline_img(:,:,j);
            scalefactor = ones(dimt,1);

            % Time loop
            for t = 1:dimt
                % Slice j at time t
                currentimgj  = dynam(:,:,j,t);
                drift_imgj = DRIFT(:,:,j);

                % rod voxel locations
                %OUT = ROD{j}.OUT;
                driftind_j = find(drift_imgj > 0);

                if(~isempty(driftind_j))
                    % signal relative to that at the second time point
                    scalefactor(t) = mean(originalimgj(driftind_j))/mean(currentimgj(driftind_j));
                end
            end

            if(~isempty(driftind_j))
                % Fit time curve of rod to poly4
                scale_fit{j} = fit((1:numel(scalefactor))',scalefactor,'poly4','Robust','on');
            end
        end

        % Second Scale images
        DYNAM       = [];
        DYNAMLV     = [];
        DYNAMNOISE  = [];
        DYNAMNONVIA = [];
        scale_index = zeros(1,dimz);

        % Time loop
        for time_index = 1:dimt
            % All slices at the time time_index
            currentimg       = dynam(:,:,:,time_index);
            % Slice loop
            for j = 1:dimz 
                % Slice j at time time_index
                currentimgj  = currentimg(:,:,j);
                drift_imgj = DRIFT(:,:,j);
                
                % rod voxel locations
                driftind_j = find(drift_imgj > 0);
                
                if(~isempty(driftind_j)) 
                    scale_index(j) = j;
                else
                    % Try to find a suitable scale factor in another slice
                    for delta = 1:dimz-1
                        if j-delta >= 1
                            drift_imgj = DRIFT(:,:,j-delta);
                            driftind_j = find(drift_imgj > 0);
                            if(~isempty(driftind_j))
                                scale_index(j) = j-delta;
                                break
                            end
                        end

                        if j+delta <= dimz
                            drift_imgj = DRIFT(:,:,j+delta);
                            driftind_j = find(drift_imgj > 0);
                            if(~isempty(driftind_j))
                                scale_index(j) = j+delta;
                                break
                            end
                        end
                    end
                end

                if(scale_index(j))
                    DRIFT_SIGNAL(time_index,j) = mean(currentimgj(driftind_j));
                    PRECORRECTED(time_index,j) = mean(currentimgj(:));
                    currentimgj = currentimgj.*scale_fit{scale_index(j)}(time_index);
                    CORRECTED(time_index,j) = mean(currentimgj(:));
                    currentimg(:,:,j) = currentimgj;
                end
            end

            DYNAM(end+1,:)   = currentimg(tumind);
            DYNAMLV(end+1,:) = currentimg(lvind);
            DYNAMNOISE(end+1)= std(single(currentimg(noiseind)));

            if(viable)
                DYNAMNONVIA(end+1,:) = currentimg(nonvia);
            end
        end

        % Plot the drift for each slice
        drift_fig = figure;

        for j = 1:dimz  
            if(scale_index)
                subplot(ceil(sqrt(dimz)), ceil(sqrt(dimz)), j)
                hold on
                    if scale_index(j)==j
                        % only plot rod signal if that is what was used for
                        % this correction
                        plot(DRIFT_SIGNAL(:,j)', 'r.')
                        % plot fit scaled by rod signal
                        plot(1./scale_fit{scale_index(j)}(1:time_index).*DRIFT_SIGNAL(2,j),'k')
                    else
                        %otherwise plot drift fit scaled by signal from 
                        %that slice
                        plot(1./scale_fit{scale_index(j)}(1:time_index).*PRECORRECTED(2,j),'k')
                    end
                    plot(CORRECTED(:,j), 'gx')
                    plot(PRECORRECTED(:,j), 'b.')
                hold off

                title(['Slice: ' num2str(j)]);
            end
        end
        saveas(drift_fig, fullfile(PathName1, [rootname '_drift.fig']));
    end
end

%% 4.5. Automatically find AIF voxels
if strcmp(aif_rr_type,'aif_auto') || strcmp(aif_rr_type,'aif_auto_static')
    % Auto finding
    dimx = size(DYNAMIC,1);
    dimy = size(DYNAMIC,2);
    dimz = dimz;
    [end_ss, aif_index, DYNAM_AUTO_AIF] = dce_auto_aif(DYNAMLV,lvind,dimx,dimy,dimz,injection_duration);
    if isempty(aif_index)
        saved_results = '';
        return;
    end
    lvind = aif_index;
    DYNAMLV = DYNAM_AUTO_AIF;
    if strcmp(aif_rr_type,'aif_auto_static')
        LV = zeros(size(TUMOR));
        LV(lvind) = blood_t1/1000; %covert from ms to sec
        % If using a static T1 value, average curve before mM fitting to
        % reduce noise
        mean_aif = mean(DYNAMLV,2);
        for n=1:size(DYNAMLV,2)
            DYNAMLV(:,n) = mean_aif;
        end
    else
        LV_temp = zeros(size(TUMOR));
        LV_temp(lvind) = LV(lvind);
        LV = LV_temp;
    end
    
    if (exist('prctile', 'file'))
        % Update the ROI plot with the new AIF roi
        aif_mask = zeros(size_image_3d);
        aif_mask(lvind)  = 1;
        figure(nn);
        % Permute x and y
        imshow3D_overlays(permute(matchimg,[2 1 3]),...
            [prctile(reshape(matchimg,1,[]),5) prctile(reshape(matchimg,1,[]),95)],...
            permute(region_mask,[2 1 3]),...
            permute(noise_mask,[2 1 3]),...
            permute(aif_mask,[2 1 3]),...
            permute(nonviable_mask,[2 1 3]),...
            permute(drift_mask,[2 1 3])...
            );
        
    % We save the ROI image as a fig, We save in the same directory as the
    % dynamic file.
    saveas(nn, fullfile(PathName1, [rootname '_image_ROI.fig']));
    else
        warning('ROI plot not updated with AIF roi due to lack of "prctile" function. You may need the MATLAB Stats toolbox.')
    end
end

%% 5. Manually select injection point, if required
if(steady_state_time == -1)
    fig_get_baseline = figure; hold on;
    plot(mean(DYNAM, 2), 'r');
    plot(mean(DYNAMLV,2), 'b');
    ylabel('a.u.');
    xlabel('image number');
    hold off;
    
    
    for i = 1:2
        disp("Getting baseline");
        title(['Select timepoint ' num2str(i) ...
            ' before injection. (i.e. Select an interval (2 points) before contrast injection to define steady state)'])
        drawnow
        figure(fig_get_baseline);
        [steady_state_time(i), ~] = ginput(1);
    end
    steady_state_time(2) = round(steady_state_time(2));
    steady_state_time(1) = round(steady_state_time(1));
    if steady_state_time(1)<1
        %No zero index in matlab
        steady_state_time(1) = 1;
    end
    start_injection = steady_state_time(2);
    [~, end_injection] = max(DYNAMLV);
    end_injection = mean(end_injection);
elseif (steady_state_time == -2)
    if ~exist('end_ss','var')
        dimx = size(DYNAMIC,1);
        dimy = size(DYNAMIC,2);
        dimz = dimz;
        end_ss = dce_auto_aif(DYNAMLV,lvind,dimx,dimy,dimz,injection_duration);
    end
    start_injection = end_ss;
    [~, end_injection] = max(DYNAMLV);
    end_injection = mean(end_injection);
    steady_state_time(2) = end_ss;
    steady_state_time(1) = 1; 
else
    %No zero index in matlab
    steady_state_time(2) = steady_state_time;
    steady_state_time(1) = 1;
end
disp(['Steady state time selected from image ' num2str(steady_state_time(1)) ...
    ' to image ' num2str(steady_state_time(2)) ' ']);

% Averaged timecurves for AIF and Tumor
RawTUM = mean(DYNAM,2);
RawLV  = mean(DYNAMLV,2);

if(viable)
    RawNONVIA = mean(DYNAMNONVIA,2);
end


%% 6.  Filter out AIF points if SNR too low

% Save the voxel IDs of the AIF pre filtering.
oldvind = lvind;
prefilter_number_aif_voxels = size(lvind,1);

snrfilter = 0;
%Dynam(time,voxels)
DYNAMLV_time_average = mean(DYNAMLV,1);
DYNAMNOISE_time_average = mean(DYNAMNOISE);
voxelSNR = DYNAMLV_time_average./DYNAMNOISE_time_average;

ind = find(voxelSNR < snr_filter);

lvind(ind) = [];
DYNAMLV(:,ind) = [];
voxelSNR_filtered = voxelSNR;
voxelSNR_filtered(ind) = [];

snrfilter = snrfilter + numel(ind);  
% for i = 1:size(DYNAM,1)
%     currentSNR = DYNAMLV(i,:)./DYNAMNOISE(i);
%     
%     ind = find(currentSNR < snr_filter);
%     
%     lvind(ind) = [];
%     DYNAMLV(:,ind) = [];
%     
%     numel(ind);
%     snrfilter = snrfilter + numel(ind);  
% end

disp(['AIF SNR filter requires average SNR > ' num2str(snr_filter)]);
disp(['AIF SNR filter has removed: ' num2str(snrfilter) ' of ' num2str(prefilter_number_aif_voxels) ' AIF voxels.']);
disp(['After filter average AIF SNR (all voxels, all time points) = ' num2str(mean(voxelSNR_filtered))]);
if prefilter_number_aif_voxels==0
    error('Error no AIF selected, or auto AIF failed to find any voxels');
elseif snrfilter>=prefilter_number_aif_voxels
    error('Error SNR filter removed all AIF points, lower SNR filter requirement');
end

disp('Average Filtered AIF T1: ')
disp(mean(LV(lvind)));

%% 7. Convert AIF/ Reference region to R1
T1LV     = LV(lvind);
T1       = T1LV;
Stotallv = DYNAMLV;


% Sss is the steady state Signal before injection
Sss      = mean(Stotallv(round(steady_state_time(1)):round(steady_state_time(2)),:),1);
Sstar    = ((1-exp(-tr./T1))./(1-cosd(fa).*exp(-tr./T1)));
Stlv     = Stotallv;%(inj(2):end,:);

for j = 1:size(Stlv,2)
    A(:,j) = 1-cosd(fa).*Sstar(j).*Stlv(:,j)./Sss(j);
    B(:,j)  = 1-Sstar(j).*Stlv(:,j)./Sss(j);
end

AB = A./B;
% return
% AB should not be less than 1. We interpolate the timeseries to clean
% this. Threshold is 0.05, ie if 5% of timepoints <1 remove voxel;

[AB, T1LV, lvind, BADspacelABv, GOODspacelABv] = cleanAB(AB, T1LV,lvind, 'AIF', min(numel(T1LV)), 0.05, quant);
if numel(T1LV)==0
    error('A_make_R1maps_func:AIFVoxelsRemoved', 'Error: all AIF voxels removed due to bad/negative values, check T1 values and AIF placement.');
end

R1tLV = double((1/tr).*log(AB));
Sss = Sss(GOODspacelABv);
Stlv = Stlv(:,GOODspacelABv);
% R1 should be real and <100/sec. We interpolate the timeseries to clean
% this. Threshold is 0.005, ie if 0.5% of timepoints have img remove voxel;

[R1tLV, T1LV, lvind, BADspacelv, GOODspacelv] = cleanR1t(R1tLV, T1LV,lvind, 'AIF', min(numel(T1LV)), 0.005, quant);
if numel(T1LV)==0
    error('A_make_R1maps_func:AIFVoxelsRemoved', 'Error: all AIF voxels removed due to bad/negative values, check T1 values and AIF placement.');
end
Sss = Sss(GOODspacelv);
Stlv = Stlv(:,GOODspacelv);

if ~isempty(T1LV)
    for j = 1:numel(T1LV)
        % Scale R1 time curve values to initial T1 values
        ScaleFactorlv = (1/T1LV(j)) - mean(R1tLV(round(steady_state_time(1)):round(steady_state_time(2)), j),1);
        R1tLV(:,j) = R1tLV(:,j) + ScaleFactorlv;
    end
else
    error("ERROR: Arterial input matrix T1LV is empty. All AIF voxels may have been removed by the cleaning steps. " + ...
        "This is most often caused by either a poor baseline (negative perhaps) and/or poor T1 values. " + ...
        "Try cutting off a data point or two in script_preferences (start_t) and/or " + ...
        "manually setting blood T1 values (blood_t1) and changing aif_rr_type to a static option.")
end

if ~quant
    % R1 curves are not real, so we re-process Sss Stlv, we keep the other
    % time streams just to keep legacy downstream code nice.
    Sss      = mean(Stotallv(round(steady_state_time(1)):round(steady_state_time(2)),:),1);
    Stlv     = Stotallv;%(inj(2):end,:);
end

%% 8. Convert Tumor ROI to R1
clear A B AB

T1TUM     = TUMOR(tumind);
T1        = T1TUM;
Stotaltum = DYNAM;

Ssstum = mean(Stotaltum(round(steady_state_time(1)):round(steady_state_time(2)),:),1);
Sstar    = ((1-exp(-tr./T1))./(1-cosd(fa).*exp(-tr./T1)));
Sttum = Stotaltum;

for j = 1:numel(T1)
    
    A(:,j) = 1-cosd(fa).*Sstar(j).*Sttum(:,j)./Ssstum(j);
    B(:,j)  = 1-Sstar(j).*Sttum(:,j)./Ssstum(j);
    
end

AB = A./B;

[AB, T1TUM, tumind, BADspaceAB, GOODspaceAB] = cleanAB(AB, T1TUM,tumind, 'Tumor', min(numel(T1TUM)), 0.7, quant);
if numel(T1TUM)==0
    error('Error all voxels removed due to bad/negative values, check ROI, T1 values, and image quality');
end
R1tTOI = double((1/tr).*log(AB));

Ssstum = Ssstum(GOODspaceAB);
Sttum = Sttum(:,GOODspaceAB);
Sstar = Sstar(GOODspaceAB);


[R1tTOI, T1TUM, tumind, BADspaceT, GOODspaceT] = cleanR1t(R1tTOI, T1TUM,tumind, 'Tumor', min(numel(T1TUM)), 0.7, quant);
if numel(T1TUM)==0
    error('Error all voxels removed due to bad/negative values, check ROI, T1 values, and image quality');
end
Ssstum = Ssstum(GOODspaceT);
Sttum = Sttum(:,GOODspaceT);
Sstar = Sstar(GOODspaceT);

if ~isempty(T1TUM)
    for j = 1:numel(T1TUM)
        % Scale Sss values to T1 values
        ScaleFactortum = (1/T1TUM(j)) - mean(R1tTOI(round(steady_state_time(1)):round(steady_state_time(2)), j),1);
        R1tTOI(:,j) = R1tTOI(:,j) + ScaleFactortum;
    end
else
    error("ERROR: Arterial input matrix T1TUM is empty. All AIF voxels may have been removed by the cleaning steps. " + ...
    "This is most often caused by either a poor baseline (negative perhaps) and/or the data not settling into " + ...
    "a steady state. Try cutting off a data point or two in script_preferences (start_t) and/or manually setting " + ...
    "blood T1 values (blood_t1) and changing aif_rr_type to a static option.")
end

n = figure; 
if quant
    subplot(423),plot(median(R1tTOI,2), 'r'), title('R1 maps pre-filtering ROI'), ylabel('sec^-1')
    subplot(424), plot(median(R1tLV,2), 'b'), title('R1 maps pre-filtering AIF'), ylabel('sec^-1')
end
%% 9. Convert to concentrations

% f.1 AIF
%if(~iterremove)
Cp = (R1tLV-repmat((1./T1LV)', [size(R1tLV, 1) 1]))./(relaxivity*(1-hematocrit));
%end
% f.2 Tumor
Ct = (R1tTOI-repmat((1./T1TUM)', [size(R1tTOI, 1) 1]))./relaxivity;
% Ct = (R1tTOI-repmat((1./T1TUM)', [size(R1tTOI, 1) 1]))./4.0;


%% 9.1 Filter Cp more, to include voxels whose max is within the 25-75% of
%% the mean, this was used for further filtering.

% injtime = find(mean(Cp,2) == max(mean(Cp,2)));
% MAXLV = max(Cp(injtime-100:injtime+100,:), [],1);
% IQR = iqr(MAXLV);
% ind = find(abs(MAXLV - mean(MAXLV)) > IQR);
% lvind(ind) = [];
% Cp(:,ind)    = [];
% disp(['Interquartile filter had filtered out: ' num2str(numel(ind)) ' voxels.']);

%% 10. Save the Ct file as a dynamic curve


CTFILE = zeros(size(dynam));

for i = 1:size(Ct,1)
    currentCT = zeros(size(dynam,1), size(dynam,2), dimz);
    currentCT(tumind) = Ct(i,:);
    CTFILE(:,:,(i-1)*dimz+[1:dimz]) = currentCT;
end

% Save as dynamic file, with the TUMOR ROI only.

CC = make_nii(CTFILE, res, [1 1 1],[],[],hdr);
if quant
    save_nii(CC, fullfile(PathName1, [rootname 'dynamicCt.nii.gz']));
end
%% 11. Save as delta R1 values (May be useful if the Contrast agent has longer correlation time).

deltaR1LV = R1tLV-repmat(mean(R1tLV(round(steady_state_time(1)):round(steady_state_time(2)), :), 1), [size(R1tLV,1) 1]);

deltaR1TOI= R1tTOI-repmat(mean(R1tTOI(round(steady_state_time(1)):round(steady_state_time(2)), :), 1), [size(R1tTOI,1) 1]);


%% 12. Plot the time curves

figure(n), 
if quant
    subplot(427),plot(median(Ct,2), 'r'), title('Ct maps pre-filtering ROI'), ylabel('mM')
    subplot(428), plot(median(Cp,2), 'b'), title('Cp maps pre-filtering AIF'), ylabel('mM')
    
    subplot(425), plot(median(deltaR1TOI,2), 'r'), title('Delta R1 ROI'), ylabel('sec^-1')
    subplot(426), plot(median(deltaR1LV,2), 'b'), title('Delta R1 AIF') , ylabel('sec^-1')
end
subplot(421), plot(RawTUM, 'r'), title('T1-weighted ROI'), ylabel('a.u.')
subplot(422), plot(RawLV, 'b'), title('T1-weighted AIF'), ylabel('a.u.')

saveas(n,fullfile(PathName1, [rootname '_timecurves.fig']));

%% 13. Setup output structure
if quant
    Adata.CTFILE = CTFILE;
end
Adata.Cp     = Cp;
Adata.Ct     = Ct;
Adata.DYNAM  = DYNAM;
Adata.DYNAMIC= DYNAMIC;
Adata.DYNAMLV= DYNAMLV;
Adata.DYNAMLV_time_average = DYNAMLV_time_average;
Adata.DYNAMNOISE = DYNAMNOISE;
Adata.DYNAMNOISE_time_average = DYNAMNOISE_time_average;
Adata.DYNAMNONVIA = DYNAMNONVIA;
% Adata.GOODspace = GOODspace;
% Adata.GOOODspacelv = GOODspacelv;
Adata.LV = LV;
Adata.NOISE = NOISE;
Adata.PathName1 = PathName1;
Adata.R1tLV = R1tLV;
Adata.R1tTOI= R1tTOI;
Adata.ScaleFactorlv = ScaleFactorlv;
Adata.ScaleFactortum= ScaleFactortum;
Adata.Sss = Sss;
Adata.Ssstum = Ssstum;
Adata.Sstar  = Sstar;
Adata.Stlv   = Stlv;
%Adata.Stotaltum = Stotaltum;
Adata.Sttum  = Sttum;
Adata.T1 = T1;
Adata.T1LV = T1LV;
Adata.T1TUM= T1TUM;
Adata.TUMOR= TUMOR;
Adata.aif_rr_type = aif_rr_type;
Adata.currentCT = currentCT;
Adata.deltaR1LV = deltaR1LV;
Adata.deltaR1TOI= deltaR1TOI;
Adata.dynam_name= dynam_name;
Adata.dynampath = dynampath;
Adata.hdr       = hdr;
Adata.hematocrit=hematocrit;
Adata.log_path  = log_path;
Adata.matchimg  = matchimg;
Adata.noiseind  = noiseind;
Adata.oldvind   = oldvind;
Adata.quant     = quant;
Adata.relaxivity= relaxivity;
Adata.res       = res;
Adata.rootname  = rootname;
Adata.sliceloc  = sliceloc;
Adata.steady_state_time = steady_state_time;
Adata.tumind    = tumind;
Adata.voxelSNR  = voxelSNR;
Adata.voxelSNR_filtered = voxelSNR_filtered;
Adata.injection_duration = injection_duration;
Adata.start_injection = start_injection;
Adata.end_injection = end_injection;
Adata.tr = tr;
Adata.fa = fa;
Adata.blood_t1 = blood_t1;

%% 14. Save the file for the next Step

saved_results = fullfile(PathName1, ['A_' rootname 'R1info.mat']);
if save_output==true
    save(saved_results, 'Adata','-v7.3');
    Opt.Input = 'file';
    try
        %mat_md5 = DataHash(saved_results, Opt);
        disp('hash disabled')
        mat_md5 = 0;
    catch
        disp('Problem using md5 hashing. Will continue');
        mat_md5 = 'error';
    end
    disp(' ')
    disp('MAT results saved to: ')
    disp(saved_results)
    disp(['File MD5 hash: ' mat_md5])
end
RUNA_vars=Adata;
    
disp(' ');
disp('Finished A');
disp(datetime)
toc
diary off;
