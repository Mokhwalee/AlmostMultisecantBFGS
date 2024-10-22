m = 1000; n = 800;
signal = 1;
sigma = [8, 20];
eig_range = 20; % Assuming eig_range is defined somewhere in your script
class_balance = 1; % Assuming class_balance is defined somewhere in your script

c_bar = exp(-linspace(0, eig_range, n))'; % each feature has different weight
if signal == 0 % low
    c = randn(n,1); % distance
elseif signal == 1 % high
    c = randn(n,1).*(1-c_bar);  
else
    disp("Warning : Set the signal value either 0 or 1")
end

% Initialize figure
figure;
hold on;

% Define colors for the plots
colors = ['b', 'r']; % Blue and Red

for i = 1:length(sigma)
    W = sigma(i)*rand(m,n).*(ones(m,1)*c_bar'); % error of the data
    b = 2*(randn(m,1) > class_balance)-1;
    A = b*c' + W;                             % data in R^{m x d}
    
    % Compute singular values
    singular_values = svd(A);
    
    % Plot singular values with transparency
    if i == 1
        bar(singular_values, 'FaceColor', colors(i), 'FaceAlpha', 0.4, 'DisplayName', 'high signal');
    else
        bar(singular_values, 'FaceColor', colors(i), 'FaceAlpha', 0.4, 'DisplayName', 'low signal');
    end
end

% Customize plot
xlabel('Singular value', 'FontSize', 22);
ylabel('Singular Value size', 'FontSize', 22);
title('Spectrum of Hessian in sensing problem', 'FontSize', 20);
legend('show');

% Set the font size of the tick labels
ax = gca;
ax.FontSize = 18;

% Set the x-axis and y-axis limits to zoom in
xlim([0, 100]); % Adjust these values as needed
ylim([0, 500]); % Adjust these values as needed

% Save the plot as a PNG file
saveas(gcf, 'spectrum_of_hessian.png');

hold off;