
% Generate Problem
clc, clf, clear; warning('off');

% Run this script file with the current directory
% pwd = /Users/moka/Documents/GitHub/AlmostMultisecantBFGS/fig'
addpath([pwd,'fig/sensing_fig']); 
addpath([pwd,'fig/sensing_fig/trajectory_fig_signal0']);
addpath([pwd,'fig/sensing_fig/trajectory_fig_signal1']);

% Get parameters
[m, n, eig_range, class_balance, logreg_eps, stepsize, p, sigma, ...
    num_iter, iter_limit, seed, signal] = get_parameter();

signal_val = [0,1];
seed_val = [1,2,3,4,5];
sigma_val = [1,5,10];
eig_range_val = [1,5,10,15,20];

for signal = signal_val                     % 0 or 1
    for sigma = sigma_val               % [1,5,10,15,20]
        for eig_range = eig_range_val   % [1,5,10,15,20]
            for seed = seed_val                     % [1,2,3,4,5]
            
                rng(seed);
    
                % ----------- Logistic Regression Problems ----------- %
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
        
                name = 'sigma'+string(sigma) + '_eigrange'+string(eig_range)+ '_seed'+string(seed);
                folder_name = 'sensing_fig/trajectory_fig_signal'+string(signal);
                saveas(gcf, fullfile(folder_name ,'eigs_'+'sigma'+string(sigma) + '_eigrange'+string(eig_range)+'.png'));
    
                % ----------- Run QN simulations ----------- %
                run_qn_sensing_problem(n, fn, grad, stepsize, num_iter, p, iter_limit, name, folder_name);
    
            end
        end
    end
end
       