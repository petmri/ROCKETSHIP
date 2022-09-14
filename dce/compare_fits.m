function compare_fits(fitting_results,background_image_path,show_original,show_ci)

% compare_fits - Analyse the fitting results from dce (D_fit_voxels_func())
% or fitting_gui
%
% Inputs:
%  fitting_results       - *.mat Results from curve fitting
%  background_image_path - path to image used to select voxels to see fit
%                          of
%  show_original         - show original unsmooted time curve
%  show_ci               - show approximate 95% confidence interval curve
%
% The script loads the data arrays generated from D_fit_voxels_func() or
% fitting_gui. Then one can visualize the fitted curves compared to the
% original time series data
% 
% Requires:
% compare_fits.m
% plot_dce_curve.m
% plot_curve.m
% FXLStep1AIF_vpcfit.m
% FXLStep1AIFcfit.m
% fxr_cfit.m
% niftitools
%
% 
% Samuel Barnes
% Caltech
% December 2013


% Load data
load(fitting_results,'fit_data','xdata');

% Display Image
background_image = load_untouch_nii(background_image_path);
background_image = double(background_image.img);
figure(1);
% Note image transpose here
slider_handle = imshow3D(permute(background_image,[2 1 3]),...
    [prctile(reshape(background_image,1,[]),5) prctile(reshape(background_image,1,[]),95)]);

% Setup callback to capture mouse clicks
set(get(gca,'Children'),'ButtonDownFcn', @mouseClick);


% -=< Mouse click callback function >=-
    function mouseClick (object, eventdata)
        coords=get(gca,'CurrentPoint');
        
        image_x = int32(coords(1, 1));
        image_y = int32(coords(1, 2));
        if slider_handle==0
            image_z = 1;
        else
            image_z = round(get(slider_handle,'Value'));
        end

        % X and Y are reversed because of the transpose above
        size_y = size(background_image,1);
        size_x = size(background_image,2);
        image_id = image_x+(image_y-1)*size_y+(image_z-1)*size_x*size_y;
        
        if sum(strcmp(fit_data.model_name,{'aif' 'aif_vp' 'fxr' 'tofts' 'ex_tofts' 'patlak' 'nested' 'tissue_uptake' '2cxm'}))
            voi = find(fit_data.tumind==image_id);
            if ~isempty(voi)
                plot_data.Ct = xdata{1}.Ct(:,voi);
                plot_data.Ct_original = xdata{1}.Ct_original(:,voi);
                plot_data.Cp = xdata{1}.Cp;
                plot_data.timer =  xdata{1}.timer;
                plot_data.x_units = 'Time (minutes)';
                plot_data.y_units =  'Concentration (mM)';
                plot_data.fit_parameters = fit_data.fitting_results(voi,:);
                plot_data.model_name = fit_data.model_name;
                plot_data.show_original = show_original;
                plot_data.show_ci = show_ci;
                plot_data.title = [fit_data.model_name ' at Voxel Location (' num2str(image_x) ',' num2str(image_y) ',' num2str(image_z) ')'];

                if strcmp(fit_data.model_name,'fxr')
                    plot_data.R1o = xdata{1}.R1o(voi);
                    plot_data.R1i = xdata{1}.R1i(voi);
                    plot_data.r1 = xdata{1}.relaxivity;
                    plot_data.fw = 0.8;
                end

                figure(2);
                plot_dce_curve(plot_data);
            end
        else
            if fit_data.fitting_results(image_id,1)>0
                % Assume it is a decay model then
                plot_data.x_values = xdata{1}.x_values;
                plot_data.y_values = squeeze(xdata{1}.y_values(image_x,image_y,image_z,:));
                plot_data.x_units = xdata{1}.x_units;
                plot_data.y_units =  xdata{1}.y_units;
                plot_data.fit_parameters = fit_data.fitting_results(image_id,:);
                plot_data.model_name = fit_data.model_name;
                plot_data.show_original = show_original;
                plot_data.show_ci = show_ci;
                plot_data.title = [fit_data.model_name ' at Voxel Location (' num2str(image_x) ',' num2str(image_y) ',' num2str(image_z) ')'];

                if (contains(fit_data.model_name,'t1_fa') || contains(fit_data.model_name,'t1_b1_fa'))
                    plot_data.tr = xdata{1}.tr;
                end
                
                figure(2);
                plot_curve(plot_data);
            end
        end
    end

end

