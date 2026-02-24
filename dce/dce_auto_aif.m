function [end_ss, aif_index, dynamic_output] = dce_auto_aif(dynamic_input, mask_index,dimx,dimy,dimz,injection_duration)

% Only calculate the end of steady state, stop here
if nargout == 1
    disp('Running auto find end baseline/steady state, searching...');
else
    disp('Running auto AIF selection, searching for AIF...');
end


% Get preferences
prefs = parse_preference_file('dce_preferences.txt',0,...
    {'autoaif_r_square_threshold' 'autoaif_end_signal_threshold' ...
    'autoaif_sobel_threshold' });
r_square_threshold = str2num(prefs.autoaif_r_square_threshold);
end_signal_threshold = str2num(prefs.autoaif_end_signal_threshold);
sobel_threshold = str2num(prefs.autoaif_sobel_threshold);
% r_square_threshold = 0.8;
% end_signal_threshold = 0.35;


% Shape data for detection
spatial_points = size(dynamic_input,2);
time_points = size(dynamic_input,1);
timer   = 0:time_points-1;
dynam_smooth = smooth(dynamic_input,25,'moving');
dynam_smooth = reshape(dynam_smooth,time_points,spatial_points);
% Scale from 0 to 1 so edge detector threshold is consistant
max_smooth = max(max(dynam_smooth));
min_smooth = min(min(dynam_smooth));
dynam_smooth = (dynam_smooth-min_smooth)./(max_smooth-min_smooth);


% Find start of injection
global_dyn = mean(dynamic_input,2);
global_smooth = smooth(global_dyn,9,'moving');
max_smooth = max(max(global_smooth));
min_smooth = min(min(global_smooth));
global_smooth = (global_smooth-min_smooth)./(max_smooth-min_smooth);
% [bw thresh]= edge(global_smooth,'sobel',0.010);
% Calculate sobel derivative look for min value (max rising edge)
% Predict that backwards to find the beginning of the rising edge
op = fspecial('sobel')/8;
x_mask = op;
bx = imfilter(global_smooth,x_mask,'replicate');
[slope, i] = min(bx);
% slope = mean(bx(i-1:i+1));
% midpoint = global_smooth(i);
% offset = midpoint/slope;
% end_ss=round(i+offset);
% offset_forward = (1-midpoint)/-slope;
% end_inject=round(i+offset_forward);

% all_rsquare(1) = 0;
end_ss = i-1;
for n=2:(i-1)
    [fitobject, gof] = fit((n:i)',global_smooth(n:i),'poly1');
%     all_rsquare(n) = gof.rsquare;
    if gof.rsquare> 0.95
        end_ss = n;
        break;
    end
end

fprintf('Found end steady state at image number: %d\n',end_ss);

% Only calculate the end of steady state, stop here
if nargout == 1
    return;
end

end_inject = end_ss + injection_duration;

figure
plot(global_smooth)
hold on;
injection = zeros(size(global_smooth));
injection(end_ss) = 1;
injection(end_inject) = 1;
plot(injection,'r');
hold off;
% plot(bw,'r');
% plot(bx,'r');
% waitforbuttonpress

dynamic_output = [];
edge_detected = zeros(spatial_points,1);
all_edges = 0;
positive_edges = 0;
% positive_edges_robust = 0;
rsquare_edges = 0;
good_fit_edges = 0;
tic
p = ProgressBar(100);
progress_interval = round(spatial_points/100);
% Hack to make xdata compatible with parfor
xdata = cell(spatial_points,1);
parfor voxel_index = 1:spatial_points

    % Edge detect with sobel filter
    [bw thresh]= edge(dynam_smooth(:,voxel_index),'sobel',sobel_threshold);
    % Look only for edges within injection window
    if max(bw(end_ss:end_inject))==1   
        all_edges = all_edges+1;
        
        
%         bw_plot = bw*0.5;     
% %         bw_plot = is_edge_positive_robust.*10+min(dynam_smooth(:,voxel_index));
%         figure(1); 
%         plot(dynam_smooth(:,voxel_index)); hold on
% %         plot(dynam_smooth_robust); hold on
%         plot(bw_plot,'r');
%         hold off;
% %         figure(2);
% %         plot(dynam_pre_smooth(:,voxel_index));
% %         thresh
% %         waitforbuttonpress
        
        
        % Calculate gradient to see if edge is positive
        op = fspecial('sobel')/8;
        x_mask = op;
        bx = imfilter(dynam_smooth(:,voxel_index),x_mask,'replicate');
        is_edge_positive = -bw.*bx;

        % Eliminate negative edges
        if max(is_edge_positive(end_ss:end_inject))>0
            positive_edges = positive_edges+1;
            % Run robust smoothing to remove more noise
            % Very slow so only do it after is has been pre selected
            % with faster algorithm
%             dynam_smooth_robust = smooth(dynam_pre_smooth(:,voxel_index),25,'rlowess');
%             max_smooth = max(dynam_smooth_robust);
%             min_smooth = min(dynam_smooth_robust);
%             dynam_smooth_robust = (dynam_smooth_robust-min_smooth)./(max_smooth-min_smooth);
% 
%             bw_robust = edge(dynam_smooth_robust,'sobel',0.007);
%             is_edge_positive_robust = -bw_robust.*bx;
%             
%             % Remove edges that fail after robust smoothing
%             if max(is_edge_positive_robust(end_ss:end_inject))>0
%                 positive_edges_robust = positive_edges_robust+1;

            % Fit voxel to biexponential and see what r^2 is
            aif_x = dynamic_input(:,voxel_index);
            % Set baseline to zero so it will fit properly
            aif_x = aif_x - mean(aif_x(1:end_ss));
            % Scale to 1
            aif_x = aif_x./max(aif_x);
            xdata{voxel_index}.Cp    = aif_x;
            xdata{voxel_index}.timer = timer';
            xdata{voxel_index}.step = [end_ss end_inject];
            % Run fit
            [aif_fitted, ~, ~, rsquare] = AIFbiexpfithelp(xdata{voxel_index}, 0);
            % Remove bad fits
            if rsquare>r_square_threshold
                rsquare_edges = rsquare_edges+1;
                if aif_fitted(end)/max(aif_fitted)<end_signal_threshold
                    edge_detected(voxel_index) = 1; 
                    good_fit_edges = good_fit_edges+1;                  
    %                 figure;
    %                 plot(timer,aif_x,'r.');
    %                 hold on;
    %                 plot(timer, aif_fitted,'b');
    %                 title(num2str(rsquare));
    %                 hold off;
    %                 waitforbuttonpress
                end
            end
%             end
        end
    end
    if mod(voxel_index,progress_interval)==0, p.progress; end
end
p.stop;
% Free memeory
clear xdata
% Report results
disp('Finished AIF search');
disp(['Found ' num2str(good_fit_edges) ' AIF voxels']);
toc
fprintf('Edges detected with Sobel filter: %d\n',all_edges);
fprintf('Rising edges: %d\n',positive_edges);
fprintf('Fit with r^2 > %.2f: %d\n',r_square_threshold,rsquare_edges);
fprintf('End signal < %.2f of max: %d\n',end_signal_threshold, good_fit_edges);

% figure(1)
% imshow(DYNAMIC(:,:,125)',[]);
% figure(2)
% imshow(reshape(edge_detected,size(DYNAMIC,1),size(DYNAMIC,2))');

aif_mask = zeros(dimx,dimy,dimz);
aif_mask(mask_index) = edge_detected;

% Plot redundant, removed
% Plot found AIF voxels on image
% figure;
% dynamic_slice = zeros(dimx,dimy,dimz);
% dynamic_slice(mask_index) = dynamic_input(end_inject,:);
% % red_mask = cat(3, ones([dimy dimx]), zeros([dimy dimx]), zeros([dimy dimx]));
% zero_mask = zeros(dimx,dimy,dimz);
% imshow3D(flip(flip(permute(dynamic_slice,[2 1 3]),1),2),...
%             [prctile(reshape(dynamic_slice,1,[]),5) prctile(reshape(dynamic_slice,1,[]),95)],...
%             zero_mask,zero_mask,flip(flip(permute(aif_mask,[2 1 3]),1),2),zero_mask);


% Generate output, list of aif voxels and associated time curves
aif_index = find(aif_mask>0);
dynamic_output = dynamic_input(:,edge_detected>0);
% Plot average of found AIF and fit
if ~isempty(aif_index)
    aif_found = mean(dynamic_output,2);
    aif_found = aif_found - mean(aif_found(1:end_ss));
    aif_found = aif_found./max(aif_found);
    xdata{1}.Cp    = aif_found;
    xdata{1}.timer = timer';
    xdata{1}.step = [end_ss end_inject];
    % Show fit
    [aif_fitted, ~, ~, rsquare] = AIFbiexpfithelp(xdata, 0);
    figure;
    plot(timer,aif_found,'r.');
    hold on;
    plot(timer, aif_fitted,'b');
    title(['r^2 = ' num2str(rsquare)]);
    hold off;
else
    disp('No AIF found');
end
       

