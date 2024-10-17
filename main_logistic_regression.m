%{
1. L-BFGS (single, two-loop) : l_bfgs_2loop.m
2. L-MS-BFGS (brute_force old version) : ms_bfgs_brute_force.m
3. L-MS-BFGS (brute-force for two-loop) : ms_bfgs_brute_force_2loop.m
4. L-MS-BFGS (extended) : ms_bfgs_extended
5. L-MS-BFGS (two-loop) : l_ms_bfgs_ours_2loop.m
6. L-MS-BFGS-mu (two-loop) : l_ms_bfgs_ours_2loop_mu.m
7. MS-BFGS-Schur-inv-mu (IEEE) : ms_bfgs_schur_inv_mu.m
%}

% Generate Problem
clc, clf, clear; warning('off');

% Get parameters
[m, n, eig_range, class_balance, logreg_eps, stepsize, p, sigma, ...
    num_iter, iter_limit, seed, signal] = get_parameter();
rng(seed);

% ----------- Logistic Regression Problems ----------- %%
[fn, grad, prob_difficulty, matrix_A, y_sol, bA] = ...
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
xlim([0, n/2])

name = 'size_'+string(m)+'by'+string(n) + '_seed_'+string(seed) + ...
        '_signal_'+string(signal) + '_sigma_'+string(sigma);

saveas(gcf, fullfile('fig/sensing_fig/' ,name+'_eigval.png'));



%% ----------- Initialization ----------- %%
B = eye(n); 
x0 = zeros(n,1); 
f0 = fn(x0); %initial function val

% get optimal solution from single bfgs
[f_optimal, traj_opt, x_opt, traj_grad] = ...
    single_bfgs_vanilla(B, x0, 0.01, 100000, fn, grad);

% Run QN methods
[f_l_bfgs_2loop, traj_l_bfgs_2loop, x_l_bfgs_2loop, traj_l_bfgs_2loop_grad, err_l_bfgs_2loop] = ...
    single_l_bfgs_2loop(x0, stepsize, num_iter, p, fn, grad, x_opt);

[f_multi_bfgs, traj_multi_bfgs, x_multi_bfgs, traj_multi_bfgs_grad, err_multi_bfgs] = ... % Vanilla
    ms_bfgs_vanilla(B, x0, stepsize, num_iter, p, fn, grad, x_opt);

[f_l_ms_bfgs_paper, traj_l_ms_bfgs_paper, x_l_ms_bfgs_paper, traj_l_ms_bfgs_paper_grad, err_l_ms_bfgs_paper] = ...
    l_ms_bfgs_paper(x0, stepsize, num_iter, p, fn, grad, x_opt); 

[f_l_ms_bfgs_2loop, traj_l_ms_bfgs_2loop, x_l_ms_bfgs_2loop, traj_l_ms_bfgs_2loop_grad, err_l_ms_bfgs_2loop] = ...    
    l_ms_bfgs_2loop(x0, stepsize, num_iter, p, fn, grad, x_opt);

[f_l_ms_bfgs_2loop_mu, traj_l_ms_bfgs_2loop_mu, x_l_ms_bfgs_2loop_mu, traj_l_ms_bfgs_2loop_mu_grad, err_l_ms_bfgs_2loop_mu] = ...    
    l_ms_bfgs_2loop_mu(x0, stepsize, num_iter, p, fn, grad, iter_limit, x_opt);

[f_ms_bfgs_schur_inv, traj_ms_bfgs_schur_inv, x_ms_bfgs_schur_inv, traj_ms_bfgs_schur_inv_grad, err_ms_bfgs_schur_inv] = ...
    ms_bfgs_schur_inv_mu(x0, stepsize, num_iter, p, fn, grad, iter_limit, x_opt);


% Error rate graph of each method
error_bfgs = [
    err_l_bfgs_2loop, ...
    err_multi_bfgs, ...
    err_l_ms_bfgs_paper, ...
    err_l_ms_bfgs_2loop, ...
    err_l_ms_bfgs_2loop_mu, ...
    err_ms_bfgs_schur_inv, ... 
    ];
%%
loglog(error_bfgs,'-O', 'MarkerSize', 3)
legend({'L-BFGS (baseline, two-loop)', ...
    'MS BFGS (baseline)', ...
    'L-MS-BFGS (paper)', ...
    'L-MS-BFGS (2-loop)', ...
    'L-MS-BFGS-Schur-inv-mu (ours)', ...
    'MS-BFGS-Schur-inv-mu (ours)', ...
    }, Location="southwest", Fontsize=14)

% Set the font size of the tick labels
ax = gca;
ax.XAxis.FontSize = 20; % Change the font size of the x-axis tick labels
ax.YAxis.FontSize = 20; % Change the font size of the y-axis tick labels

xlabel("Iteration", FontSize=30)
ylabel("Error", 'Rotation', 0, Fontsize=25)
title('Logistic Regression Loss (log-log)', FontSize=26)


%% -------- trajectory -------- %%
trajectory_bfgs = [
    traj_l_bfgs_2loop, ...
    traj_multi_bfgs, ...
    traj_l_ms_bfgs_paper, ...
    traj_l_ms_bfgs_2loop, ...
    traj_l_ms_bfgs_2loop_mu, ...
    traj_ms_bfgs_schur_inv  
    ];


%% -------- graph : f-f* -------- %%
graph = trajectory_bfgs - f_optimal;
loglog(graph,'-O', 'MarkerSize', 3)
legend({'L-BFGS (baseline, two-loop)', ...
    'MS BFGS (baseline)', ...
    'L-MS-BFGS (paper)', ...
    'L-MS-BFGS (2-loop)', ...
    'L-MS-BFGS-Schur-inv-mu (ours)', ...
    'MS-BFGS-Schur-inv-mu (ours)', ...
    }, Location="southwest", Fontsize=14)

% Set the font size of the tick labels
ax = gca;
ax.XAxis.FontSize = 20; % Change the font size of the x-axis tick labels
ax.YAxis.FontSize = 20; % Change the font size of the y-axis tick labels

xlabel("Iteration", FontSize=30)
ylabel("f(x)", 'Rotation', 0, Fontsize=25)
title('Logistic Regression Loss (log-log)', FontSize=26)

% Save the figure as a PNG file
saveas(gcf, fullfile('fig/sensing_fig/' ,name+'_obj.png'));


% Gradient Trajectory graph 
    trajectory_bfgs_grad = [
    traj_l_bfgs_2loop_grad, ...
    traj_multi_bfgs_grad, ...
    traj_l_ms_bfgs_paper_grad, ...
    traj_l_ms_bfgs_2loop_grad, ...
    traj_l_ms_bfgs_2loop_mu_grad, ...
    traj_ms_bfgs_schur_inv_grad  
    ];

    loglog(trajectory_bfgs_grad,'-O', 'MarkerSize', 3)
    legend({'L-BFGS (baseline, two-loop)', ...
        'MS BFGS (baseline)', ...
        'L-MS-BFGS (paper)', ...
        'L-MS-BFGS (2-loop)', ...
        'L-MS-BFGS-Schur-inv-mu (ours)', ...
        'MS-BFGS-Schur-inv-mu (ours)', ...
        }, Location="southwest", Fontsize=13)

    % Set the font size of the tick labels
    ax = gca;
    ax.XAxis.FontSize = 20; % Change the font size of the x-axis tick labels
    ax.YAxis.FontSize = 20; % Change the font size of the y-axis tick labels
    
    xlabel("Iteration", FontSize=30)
    ylabel("||\nabla f(x)||_2^2", 'Rotation', 90, Fontsize=25)
    title('Logistic Regression(log-log)', FontSize=26)

    % Save the figure as a PNG file
    saveas(gcf, fullfile('fig/sensing_fig/' ,name+'_grad.png'));

    % save the workspace
    save(fullfile('fig/sensing_fig/', name+'_workspace.mat'));

    %% sign diff btw optimal value x and the other approx sols
    sum(sign(x_opt) ~= sign(x_l_bfgs_2loop))
    sum(sign(x_opt) ~= sign(x_multi_bfgs))
    sum(sign(x_opt) ~= sign(x_l_ms_bfgs_paper))
    sum(sign(x_opt) ~= sign(x_l_ms_bfgs_2loop))
    sum(sign(x_opt) ~= sign(x_l_ms_bfgs_2loop_mu))
    sum(sign(x_opt) ~= sign(x_ms_bfgs_schur_inv))

    %%
    f_optimal
    f_l_ms_bfgs_2loop
    f_ms_bfgs_schur_inv
    f_l_ms_bfgs_2loop_mu
    f_l_ms_bfgs_paper
    f_multi_bfgs
    f_l_bfgs_2loop

    