% Generate Problem
clc, clf, clear; warning('off');

% Get parameters
[m, n, eig_range, class_balance, logreg_eps, stepsize, p, sigma, ...
    num_iter, iter_limit, seed, signal] = get_parameter();
rng(seed);

sigma_val = [1,5,10,15,20];
eig_range_val = [1,5,10,15,20];
signal_val = [0,1];
asf
for signal = signal_val
    for sigma = sigma_val
        for eig_range = eig_range_val

            % ----------- Logistic Regression Problems ----------- %%
            [fn, grad, prob_difficulty, matrix_A, y_sol] = ...
             logistic_regression(m, n, seed, sigma, class_balance, logreg_eps, eig_range, signal);
    
            % -------- Plot histogram of singular values of matrix_A -------- %%
            singular_values = svd(matrix_A);
            bar(singular_values); % Adjust 'NumBins' as needed
            xlabel('Descending order of each singular value', 'FontSize', 22);
            ylabel('Singular Value', 'FontSize', 22);
            title('Bar Graph of the Singular Values', 'FontSize', 20);
    
            % Set the font size of the tick labels
            ax = gca;
            ax.XAxis.FontSize = 20; % Change the font size of the x-axis tick labels
            ax.YAxis.FontSize = 20; % Change the font size of the y-axis tick labels
    
            % Set the ceiling of the y-axis
            ylim([0, singular_values(1)]); % Adjust the upper limit as needed
            xlim([-1, n])
    
            name = 'eigrange'+string(eig_range) + 'sigma'+string(sigma);
            folder_name = 'fig/trajectory_fig_signal'+string(signal)+'/';
            saveas(gcf, fullfile(folder_name ,name+'.png'));

        end
    end
end
       