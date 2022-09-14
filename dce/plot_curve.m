function plot_curve(plot_data)
fit_parameters = plot_data.fit_parameters;
if fit_parameters(4)==-1, fit_parameters(4)=nan; end
if fit_parameters(5)==-1, fit_parameters(5)=nan; end

x_min = min(plot_data.x_values);
x_max = max(plot_data.x_values);
xdata_curve = x_min:(x_max-x_min)/100:x_max;

if(strcmp(plot_data.model_name,'t2_exponential') || ...
       strcmp(plot_data.model_name,'t2_linear_weighted') || ...
       strcmp(plot_data.model_name,'t2_linear_simple') || ...
       strcmp(plot_data.model_name,'t2_linear_fast'))
    fit_curve = exp(-xdata_curve./fit_parameters(1)).*fit_parameters(2);
    fit_curve_low = exp(-xdata_curve./fit_parameters(4)).*fit_parameters(2);
    fit_curve_high = exp(-xdata_curve./fit_parameters(5)).*fit_parameters(2);
elseif(strcmp(plot_data.model_name,'ADC_linear_weighted') || ...
       strcmp(plot_data.model_name,'ADC_linear_simple') || ...
       strcmp(plot_data.model_name,'ADC_linear_fast') || ...
       strcmp(plot_data.model_name,'ADC_exponential'))
    fit_curve = exp(-xdata_curve.*fit_parameters(1)).*exp(fit_parameters(2));
    fit_curve_low = exp(-xdata_curve.*fit_parameters(4)).*exp(fit_parameters(2));
    fit_curve_high = exp(-xdata_curve.*fit_parameters(5)).*exp(fit_parameters(2));
elseif(strcmp(plot_data.model_name,'t1_tr_fit'))
    fit_curve = (1-exp(-xdata_curve./fit_parameters(1))).*fit_parameters(2);
    fit_curve_low = (1-exp(-xdata_curve./fit_parameters(4))).*fit_parameters(2);
    fit_curve_high = (1-exp(-xdata_curve./fit_parameters(5))).*fit_parameters(2);
elseif(strcmp(plot_data.model_name,'t1_fa_fit'))
    fit_curve = fit_parameters(2).*( (1-exp(-plot_data.tr/fit_parameters(1))).*sind(xdata_curve) )./( 1-exp(-plot_data.tr/fit_parameters(1)).*cosd(xdata_curve) );
    fit_curve_low = fit_parameters(2).*( (1-exp(-plot_data.tr/fit_parameters(4))).*sind(xdata_curve) )./( 1-exp(-plot_data.tr/fit_parameters(4)).*cosd(xdata_curve) );
    fit_curve_high = fit_parameters(2).*( (1-exp(-plot_data.tr/fit_parameters(5))).*sind(xdata_curve) )./( 1-exp(-plot_data.tr/fit_parameters(5)).*cosd(xdata_curve) );
elseif(strcmp(plot_data.model_name,'t1_fa_linear_fit'))
%     y_lin = si./sin(pi/180*parameter);
%     x_lin = si./tan(pi/180*parameter);
    fit_curve = fit_parameters(2)./(1./sind(xdata_curve)-exp(-plot_data.tr/fit_parameters(1))./tand(xdata_curve));
    fit_curve_low = fit_parameters(2)./(1./sind(xdata_curve)-exp(-plot_data.tr/fit_parameters(4))./tand(xdata_curve));
    fit_curve_high = fit_parameters(2)./(1./sind(xdata_curve)-exp(-plot_data.tr/fit_parameters(5))./tand(xdata_curve));
elseif(strcmp(plot_data.model_name,'t1_ti_exponential_fit'))
    fit_curve = abs( fit_parameters(2).*(1-2.*exp(-xdata_curve./fit_parameters(1))-exp(-xdata_curve./fit_parameters(1)) ) );
    fit_curve_low = abs( fit_parameters(2).*(1-2.*exp(-xdata_curve./fit_parameters(4))-exp(-xdata_curve./fit_parameters(4)) ) );
    fit_curve_high = abs( fit_parameters(2).*(1-2.*exp(-xdata_curve./fit_parameters(5))-exp(-xdata_curve./fit_parameters(5)) ) );  
elseif(strcmp(plot_data.model_name,'t2_exponential_plus_c'))
    fit_curve = exp(-xdata_curve./fit_parameters(1)).*fit_parameters(2)+fit_parameters(7);
    fit_curve_low = exp(-xdata_curve./fit_parameters(4)).*fit_parameters(2)+fit_parameters(8);
    fit_curve_high = exp(-xdata_curve./fit_parameters(5)).*fit_parameters(2)+fit_parameters(9);
elseif(strcmp(plot_data.model_name,'t1_b1_fa_fit'))
    fit_curve = fit_parameters(2).*( (1-exp(-plot_data.tr/fit_parameters(1))).*sind(fit_parameters(7)*xdata_curve) )./( 1-exp(-plot_data.tr/fit_parameters(1)).*cosd(fit_parameters(7)*xdata_curve) );
    fit_curve_low = fit_parameters(2).*( (1-exp(-plot_data.tr/fit_parameters(4))).*sind(fit_parameters(8)*xdata_curve) )./( 1-exp(-plot_data.tr/fit_parameters(4)).*cosd(fit_parameters(8)*xdata_curve) );
    fit_curve_high = fit_parameters(2).*( (1-exp(-plot_data.tr/fit_parameters(5))).*sind(fit_parameters(9)*xdata_curve) )./( 1-exp(-plot_data.tr/fit_parameters(5)).*cosd(fit_parameters(9)*xdata_curve) );
elseif(strcmp(plot_data.model_name, 'user_input'))
%     [PATHSTR,NAME,~] = fileparts(userfile);
%     userFN = str2func(NAME);
%     [cf_, gof, output] = userFN(parameter(ok_), si(ok_));
elseif(strcmp(plot_data.model_name,'none'))
end

% Sort fitted line incase they are out of order
[xsorted, i_sort] = sort(plot_data.x_values);

p = plot(xsorted,plot_data.y_values(i_sort),...
    xdata_curve,fit_curve);
set(p(1),'Marker','.','MarkerSize',14,'LineStyle','none');
set(p(1),'Color','k');
set(p(2),'Color','b');

if plot_data.show_ci
    hold on
    p_ci = plot(xdata_curve,fit_curve_low,...
        xdata_curve,fit_curve_high);
    set(p_ci(1),'Color',[0.5 0.5 0.6]);
    set(p_ci(2),'Color',[0.5 0.5 0.6]);
end
hold off

plot_limits = axis;
% Estimate Error
parameter1_error = mean([abs(fit_parameters(1)-fit_parameters(4)) abs(fit_parameters(1)-fit_parameters(5))]);
    
if(strcmp(plot_data.model_name,'t2_exponential') || ...       
        strcmp(plot_data.model_name,'t2_linear_fast') || ...
        strcmp(plot_data.model_name,'t2_linear_simple') || ...
        strcmp(plot_data.model_name,'t2_linear_weighted'))
    plot_str(1) = {[' T_2 = ' num2str(fit_parameters(1),3) '±' num2str(parameter1_error,2)]};
    plot_str(2) = {[' r^2 = ' num2str(fit_parameters(3),4)]};
    plot_str(3) = {[' residual = ' num2str(fit_parameters(6),3)]};
elseif(strcmp(plot_data.model_name,'ADC_linear_weighted') || ...
        strcmp(plot_data.model_name,'ADC_exponential') || ...
        strcmp(plot_data.model_name,'ADC_linear_simple') || ...
        strcmp(plot_data.model_name,'ADC_linear_fast'))
    plot_str(1) = {[' ADC = ' num2str(fit_parameters(1),2) '±' num2str(parameter1_error,2)]};
    plot_str(2) = {[' r^2 = ' num2str(fit_parameters(3),2)]};
    plot_str(3) = {[' residual = ' num2str(fit_parameters(6))]};
elseif(strcmp(plot_data.model_name,'t1_tr_fit') ||...
        strcmp(plot_data.model_name,'t1_ti_exponential_fit') || ...
        strcmp(plot_data.model_name,'t1_fa_fit') ||...
        strcmp(plot_data.model_name,'t1_fa_linear_fit'))
    plot_str(1) = {[' T_1 = ' num2str(fit_parameters(1),4) '±' num2str(parameter1_error,2)]};
    plot_str(2) = {[' r^2 = ' num2str(fit_parameters(3),2)]};
    plot_str(3) = {[' residual = ' num2str(fit_parameters(6))]};
elseif( strcmp(plot_data.model_name,'t1_b1_fa_fit') )
    plot_str(1) = {[' T_1 = ' num2str(fit_parameters(1),4) '±' num2str(parameter1_error,2)]};
    plot_str(2) = {[' B_1 = ' num2str(fit_parameters(7),3)]};
    plot_str(3) = {[' r^2 = ' num2str(fit_parameters(3),2)]};
    plot_str(4) = {[' residual = ' num2str(fit_parameters(6))]};
elseif(strcmp(plot_data.model_name,'t2_exponential_plus_c'))
    plot_str(1) = {[' T_2 = ' num2str(fit_parameters(1),3) '±' num2str(parameter1_error,2)]};
    plot_str(2) = {[' r^2 = ' num2str(fit_parameters(3),4)]};
    plot_str(3) = {[' residual = ' num2str(fit_parameters(6),3)]};
    plot_str(4) = {[' C = ' num2str(fit_parameters(7),3)]};
elseif(strcmp(plot_data.model_name, 'user_input'))
    plot_str(1) = {[' ']};
elseif(strcmp(plot_data.model_name,'none'))
    plot_str(1) = {[' ']};
else
    plot_str(1) = {[' ']};
end
text(plot_limits(1),plot_limits(3),plot_str,...
    'Color', 'black',...
    'VerticalAlignment','bottom',...
    'HorizontalAlignment','left');
title(plot_data.title,'Interpreter','none');
xlabel(plot_data.x_units);
ylabel(plot_data.y_units);