
% Generate Problem
clc, clf, clear; warning('off');

% Run this script file from the below directory
% pwd = /Users/moka/Documents/GitHub/AlmostMultisecantBFGS'
%addpath([pwd,'fig/sensing_fig/trajectory_fig_signal0']);
%addpath([pwd,'fig/sensing_fig/trajectory_fig_signal1']);

% Get parameters : gonna use m, n, p, stepsize, num_iter, iter_limit
[m, n, eig_range, class_balance, logreg_eps, stepsize, p, sigma, ...
    num_iter, iter_limit, seed, signal] = get_parameter();

%seed_val = [1,2,3,4,5,6,7,8,9,10]; % seed number
seed_val = [];
for i=1:10, seed_val = [seed_val, randi(100)]; end
sigma_val = [0.5, 1, 5, 10, 20]; eig_range_val = [1,5,10,15,20];
signal_val = [0, 1];
mn_list = ["m_smaller_n"]; %["m_bigger_n" "m_smaller_n"];

for mn = mn_list
    %if mn == mn_list(1), m=1000; n=200; end
    %if mn == mn_list(2), m=100; n=200; end
    if mn == mn_list(1), m=100; n=200; end

    for signal = signal_val                 % 0 or 1
        for sigma = sigma_val               % [1,5,10,15,20]
            for eig_range = eig_range_val   % [1,5,10,15,20]
                
                iter = 0;
                for seed = seed_val         % [1,2,3,4,5]
                    
                    rng(seed);

        
                    % ----------- Logistic Regression Problems ----------- %
                    [fn, grad, prob_difficulty, matrix_A, y_sol] = ...
                     logistic_regression(m, n, seed, sigma, class_balance, logreg_eps, eig_range, signal);
            
                    % -------- Plot Bar Graph of singular values of matrix_A -------- %%
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
            
                    name = 'signal_'+string(signal)+'_sigma_'+string(sigma) + '_eigrange_'+string(eig_range)+ '_seed_'+string(iter);
                    folder_name = 'fig/sensing_fig/' + mn;
                    if seed == seed_val(1)
                        saveas(gcf, fullfile(folder_name ,'bargraph_'+ name+'.png'));
                    end
                     
                    % Update the plot at every iteration
                    % drawnow;
                    
                    % ----------- Run QN simulations ----------- %
                    [trajectories, trajectory_grad_square] = ...
                        run_qn_sensing_problem(n, fn, grad, stepsize, num_iter, p, iter_limit, name, folder_name);
                    
                    % --- Statistics - iteration by threshold --- %
                    trajectory_grad_square(isnan(trajectory_grad_square)) = Inf;
                    [fcn_iter_cell, grad_iter_cell] = get_statistics(trajectories, trajectory_grad_square);
                    
                    if iter == 0
                        avg_cell = fcn_iter_cell;
                        grad_avg_cell = grad_iter_cell;
                    else
                        % Vertically concatenate the new data to the existing cells
                        avg_cell = cellfun(@(x, y) vertcat(x, y), avg_cell, fcn_iter_cell, 'UniformOutput', false);
                        grad_avg_cell = cellfun(@(x, y) vertcat(x, y), grad_avg_cell, grad_iter_cell, 'UniformOutput', false);
                    end
                    
                    % Compute statistics at the last seed
                    if seed == seed_val(end)
                        % Get the statistics from avg_cell and grad_avg_cell by column in each cell
                        iter_avg = cellfun(@(x) mean(x), avg_cell, 'UniformOutput', false);
                        iter_std = cellfun(@(x) std(x), avg_cell, 'UniformOutput', false);

                        grad_avg = cellfun(@(x) mean(x), grad_avg_cell, 'UniformOutput', false);
                        grad_std = cellfun(@(x) std(x), grad_avg_cell, 'UniformOutput', false);

                        % Round up to 3 decimal places
                        %iter_avg = cellfun(@(x) round(x, 3), iter_avg, 'UniformOutput', false);
                        %grad_avg = cellfun(@(x) round(x, 3), grad_avg, 'UniformOutput', false);
                        %iter_std = cellfun(@(x) round(x, 3), iter_std, 'UniformOutput', false);
                        %grad_std = cellfun(@(x) round(x, 3), grad_std, 'UniformOutput', false);

                        % method : L-BFGS, MS-BFGS, L-MS-BFGS(paper), L-MS-BFGS(2loop), L-MS-BFGS-mu(ours), MS-BFGS-mu(ours)
                        % save the statistics to a file (each cell is method)
                        method = ["L-BFGS", "MS-BFGS", "L-MS-BFGS(paper)", "L-MS-BFGS(2loop)", "L-MS-BFGS-mu(ours)", "MS-BFGS-mu(ours)"];
                        excel_name = 'm'+string(m)+'n'+string(n)+'p'+string(p)+'signal_'+string(signal)+'_sigma_'+string(sigma) + '_eigrange_'+string(eig_range);
                        file_name = 'fig/sensing_fig/' + mn + '/statistics/stat_' + excel_name + '.xlsx';

                        % Initialize cell arrays to hold the data
                        data = {};

                        % Write the header
                        data = [data; {'Method', 'Metric', 'Value'}];

                        % Populate the cell arrays with data
                        for i = 1:length(method)
                            data = [data; {method(i), 'iter_avg', iter_avg{i}}];
                            data = [data; {method(i), 'iter_std', iter_std{i}}];
                            data = [data; {method(i), 'grad_avg', grad_avg{i}}];
                            data = [data; {method(i), 'grad_std', grad_std{i}}];
                        end

                        % Convert the cell array to a table
                        data_table = cell2table(data(2:end,:), 'VariableNames', data(1,:));

                        % Write the table to an Excel file
                        writetable(data_table, file_name);
                    end

                    iter = iter + 1;

                end
                
            end
        end
    end
end
       