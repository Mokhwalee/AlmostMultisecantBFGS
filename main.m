%{
Single BFGS comparison
debugging script on Mar 4

1. Vanilla Multi BFGS
2. Variations of MS-BFGS(Symm, PSD, Schur, W-Schur)
3. L-MS-BFGS(paper)
4. L-MS-BFGS Extended
5. L-MS-BFGS Brute Force
6. L-MS-BFGS Ours (new format, overleaf)
7. L-MS-BFGS Schur
%}

% Print the current working directory
disp(['Current working directory: ', pwd])

% Add the objective_function folder to the MATLAB path
addpath(fullfile(pwd, 'objective_function'));

% Generate Problem
clc, clf, clear; warning('off')

% Get parameters
[m, n, eig_range, class_balance, logreg_eps, stepsize, p, sigma, num_iter, iter_limit, seed, signal] = get_parameter();
rng(seed);

% ----------- Problems ----------- %%
% (1) Quadratic
%[fn, grad] = quadratic(m,n);

% (2) High/low signal Logreg
[fn, grad, prob_difficulty, matrix_A, y_sol] = ...
    logistic_regression(m, n, seed, sigma, class_balance, logreg_eps, eig_range, signal);

% (3) Neural Network (TBA)


% ----------- Initialization ----------- %%
B = eye(n); 
x0 = zeros(n,1); 
f0 = fn(x0); %initial function val


%% -------- Get trajectory of the each method -------- %%

%% Optimal with small step-size (gradient/hessian flow)
% Solution : Single BFGS (baseline)
[f_optimal, traj_opt, x_opt] = single_bfgs(B, x0, 0.001, 5000, fn, grad);


%% Limited Memory single BFGS
% paper version 
[f_l_bfgs, traj_l_bfgs, x_l_bfgs] = ...
    l_bfgs_paper(x0, stepsize, num_iter, p, fn, grad);

% two-loop version
[f_l_bfgs2, traj_l_bfgs2, x_l_bfgs2] = ...
    l_bfgs_2loop(x0, stepsize, num_iter, p, fn, grad);


%% MS BFGS (baseline)
[f_multi_bfgs, traj_multi_bfgs, x_multi_bfgs] = ... % Vanilla
    ms_bfgs(B, x0, stepsize, num_iter, p, fn, grad);


%%  -------- Limited MS BFGS -------- %%
% Limited MS BFGS (paper, algorithm 3.1)
[f_l_ms_bfgs_paper, traj_l_ms_bfgs_paper, x_l_ms_bfgs_paper] = ...
    l_ms_bfgs_paper(x0, stepsize, num_iter, p, fn, grad); 

% Limited MS BFGS (Ours)
[f_l_ms_bfgs_ours_paper, traj_l_ms_bfgs_ours_paper, x_l_ms_bfgs_ours_paper] = ...    
    l_ms_bfgs_ours_paper(x0, stepsize, num_iter, p, fn, grad);

[f_l_ms_bfgs_ours_2loop, traj_l_ms_bfgs_ours_2loop, x_l_ms_bfgs_ours_2loop] = ...    
    l_ms_bfgs_ours_2loop(x0, stepsize, num_iter, p, fn, grad);
    

%% -------- trajectory -------- %%
trajectory_bfgs = [traj_l_bfgs, ...
    traj_l_bfgs2, ...
    traj_multi_bfgs, ...
    traj_l_ms_bfgs_paper, ...
    traj_l_ms_bfgs_ours_paper, ...
    traj_l_ms_bfgs_ours_2loop, ...
    ];


%% -------- graph : f-f* -------- %%
graph = trajectory_bfgs - f_optimal;
loglog(graph,'-O', 'MarkerSize', 3)
legend({'L-BFGS(paper)', ...
    'L-BFGS(two-loop)', ...
    'MS BFGS(baseline)', ...
    'L-MS-BFGS(paper)', ...
    'L-MS-BFGS(ours)', ...
    }, Location="southwest", Fontsize=16)

xlabel("Iteration", FontSize=30)
ylabel("f(x)", 'Rotation', 0, Fontsize=30)
title('Logistic Regression Loss (log-log plot)', FontSize=26)

% Save the figure as a PNG file
saveas(gcf, fullfile('fig' ,'temp_figure.png'))








