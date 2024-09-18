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

% add paths to sub-folders
addpath([pwd,'/parameter']);
addpath([pwd,'/obj_fcn']);
addpath([pwd,'/qn_script']);
addpath([pwd,'/get_hessian']);
addpath([pwd,'/fig']);


% Get parameters
[m, n, eig_range, class_balance, logreg_eps, stepsize, p, sigma, ...
    num_iter, iter_limit, seed, signal] = get_parameter();
rng(seed);

% ----------- Logistic Regression Problems ----------- %%
[fn, grad, prob_difficulty, matrix_A, y_sol] = ...
    logistic_regression(m, n, seed, sigma, class_balance, logreg_eps, eig_range, signal);

% ----------- Initialization ----------- %%
B = eye(n); 
x0 = zeros(n,1); 
f0 = fn(x0); %initial function val

% -------- Get trajectory of the each method -------- %
% Optimal with small step-size (gradient/hessian flow) : Single BFGS (baseline)
[f_optimal, traj_opt, x_opt] = single_bfgs_vanilla(B, x0, 0.01, 10000, fn, grad);

% Single L-BFGS two-loop version
[f_l_bfgs_2loop, traj_l_bfgs_2loop, x_l_bfgs_2loop] = ...
    l_bfgs_2loop(x0, stepsize, num_iter, p, fn, grad);

%% MS BFGS (baseline)
[f_multi_bfgs, traj_multi_bfgs, x_multi_bfgs] = ... % Vanilla
    ms_bfgs_vanilla(B, x0, stepsize, num_iter, p, fn, grad);


%%  -------- Limited MS BFGS -------- %%
% Limited MS BFGS (paper, algorithm 3.1)
[f_l_ms_bfgs_paper, traj_l_ms_bfgs_paper, x_l_ms_bfgs_paper] = ...
    l_ms_bfgs_paper(x0, stepsize, num_iter, p, fn, grad); 

[f_l_ms_bfgs_2loop, traj_l_ms_bfgs_2loop, x_l_ms_bfgs_2loop] = ...    
    l_ms_bfgs_2loop(x0, stepsize, num_iter, p, fn, grad);

[f_l_ms_bfgs_2loop_mu, traj_l_ms_bfgs_2loop_mu, x_l_ms_bfgs_2loop_mu] = ...    
    l_ms_bfgs_2loop_mu(x0, stepsize, num_iter, p, fn, grad, iter_limit);

[f_ms_bfgs_schur_inv, traj_ms_bfgs_schur_inv, x_ms_bfgs_schur_inv] = ...
    ms_bfgs_schur_inv_mu(x0, stepsize, num_iter, p, fn, grad, iter_limit);


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
    'L-MS-BFGS-mu (ours)', ...
    'MS-BFGS-Schur-inv-mu (IEEE)', ...
    }, Location="southwest", Fontsize=14)

xlabel("Iteration", FontSize=30)
ylabel("f(x)", 'Rotation', 0, Fontsize=30)
title('Logistic Regression Loss (log-log plot)', FontSize=26)

% Save the figure as a PNG file
saveas(gcf, fullfile('fig' ,'temp_figure.png'))








