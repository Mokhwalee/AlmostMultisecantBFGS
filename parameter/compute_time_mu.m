% Define the ranges for m, n, and L
m_values = [1000000, 10000000];
n_values = [1000, 10000, 100000, 1000000, 10000000];
L_values = [5, 10, 15];
iter_limit = 10;
num_runs = 30;

% Initialize a cell array to store the results
results = {};

% Loop over the values of m, n, and L
for m = m_values
    for n = n_values
        for L = L_values
            elapsed_times = zeros(1, num_runs);
            
            for run = 1:num_runs
                % Start timing
                tic;
                
                % Generate random matrices
                W = randn(L);
                D1 = randn(m, L);
                D2 = randn(m, L);
                
                % Call the getmu function
                mu = getmu(W, D1, D2, iter_limit);
                
                % End timing and get elapsed time
                elapsed_times(run) = toc;
            end
            
            % Compute average and standard deviation
            avg_time = mean(elapsed_times);
            std_time = std(elapsed_times);
            
            % Store the results
            results = [results; {m, n, L, avg_time, std_time}];
        end
    end
end

%% Convert the results to a table
results_table = cell2table(results, 'VariableNames', {'m', 'n', 'L', 'AvgElapsedTime', 'StdElapsedTime'});

% Display the results
disp(results_table);

% Write the results to an Excel file
file_name = 'computation_times.xlsx';
writetable(results_table, file_name);