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
addpath([pwd,'/data']);
addpath([pwd,'/obj_fcn']);
addpath([pwd,'/qn_script']);
addpath([pwd,'/get_hessian']);
addpath([pwd,'/parameter']);
addpath([pwd,'/fig']);

% ----------- Logistic Regression Problems ----------- %%
num_classes = 3;
[A, b, At, bt] = get_mnist_data_multi(num_classes);
[fn, grad, m, n] = logistic_regression_multi_classification(A, b);

% ----------- Initialization ----------- %%
stepsize = 0.1;
num_iter = 200;
iter_limit = 10;
p = 5;

B = eye(n); 
x0 = zeros(n,1); 
f0 = fn(x0); %initial function val

% -------- Get trajectory of the each method -------- %
% Optimal with small step-size (gradient/hessian flow) : Single BFGS (baseline)
%[f_optimal, traj_opt, x_opt] = single_bfgs_vanilla(B, x0, 0.01, 10000, fn, grad);

%% Calculate training loss and misclassification rate
lambda = 0.0001;
theta_ls = (A' * A + lambda * eye(size(A, 2))) \ (A' * b); % Solution
y_est = A * theta_ls;       % Compute y_est for training data

% Single L-BFGS two-loop version
[f_l_bfgs_2loop, traj_l_bfgs_2loop, x_l_bfgs_2loop] = ...
    l_bfgs_2loop(x0, stepsize, num_iter, p, fn, grad);
y_l_bfgs_2loop = A * x_l_bfgs_2loop; 
train_loss_l_bfgs_2loop = get_loss(y_est, y_l_bfgs_2loop);
train_misclassify_l_bfgs_2loop = get_misclassify(y_est, y_l_bfgs_2loop);


%% MS BFGS (baseline)
[f_multi_bfgs, traj_multi_bfgs, x_multi_bfgs] = ... % Vanilla
    ms_bfgs_vanilla(B, x0, stepsize, num_iter, p, fn, grad);
y_multi_bfgs = A * x_multi_bfgs;
train_loss_multi_bfgs = get_loss(y_est, y_multi_bfgs);
train_misclassify_multi_bfgs = get_misclassify(y_est, y_multi_bfgs);


%%  -------- Limited MS BFGS -------- %%
% Limited MS BFGS (paper, algorithm 3.1)
[f_l_ms_bfgs_paper, traj_l_ms_bfgs_paper, x_l_ms_bfgs_paper] = ...
    l_ms_bfgs_paper(x0, stepsize, num_iter, p, fn, grad); 
y_l_ms_bfgs_paper = A * x_l_ms_bfgs_paper;
train_loss_l_ms_bfgs_paper = get_loss(y_est, y_l_ms_bfgs_paper);
train_misclassify_l_ms_bfgs_paper = get_misclassify(y_est, y_l_ms_bfgs_paper);


[f_l_ms_bfgs_2loop, traj_l_ms_bfgs_2loop, x_l_ms_bfgs_2loop] = ...    
    l_ms_bfgs_2loop(x0, stepsize, num_iter, p, fn, grad);
y_l_ms_bfgs_2loop = A * x_l_ms_bfgs_2loop;
train_loss_l_ms_bfgs_2loop = get_loss(y_est, y_l_ms_bfgs_2loop);
train_misclassify_l_ms_bfgs_2loop = get_misclassify(y_est, y_l_ms_bfgs_2loop);


[f_l_ms_bfgs_2loop_mu, traj_l_ms_bfgs_2loop_mu, x_l_ms_bfgs_2loop_mu] = ...    
    l_ms_bfgs_2loop_mu(x0, stepsize, num_iter, p, fn, grad, iter_limit);
y_l_ms_bfgs_2loop_mu = A * x_l_ms_bfgs_2loop_mu;
train_loss_l_ms_bfgs_2loop_mu = get_loss(y_est, y_l_ms_bfgs_2loop_mu);
train_misclassify_l_ms_bfgs_2loop_mu = get_misclassify(y_est, y_l_ms_bfgs_2loop_mu);


[f_ms_bfgs_schur_inv, traj_ms_bfgs_schur_inv, x_ms_bfgs_schur_inv] = ...
    ms_bfgs_schur_inv_mu(x0, stepsize, num_iter, p, fn, grad, iter_limit);
y_ms_bfgs_schur_inv = A * x_ms_bfgs_schur_inv;
train_loss_ms_bfgs_schur_inv = get_loss(y_est, y_ms_bfgs_schur_inv);
train_misclassify_ms_bfgs_schur_inv = get_misclassify(y_est, y_ms_bfgs_schur_inv);


%% -------- trajectory -------- %%
yt_est = At * theta_ls;             % Compute yt_est for test data
test_loss = get_loss(bt, yt_est);   % Test dataset loss value

trajectory_loss = [
    traj_l_bfgs_2loop, ...
    traj_multi_bfgs, ...
    traj_l_ms_bfgs_paper, ...
    traj_l_ms_bfgs_2loop, ...
    traj_l_ms_bfgs_2loop_mu, ...
    traj_ms_bfgs_schur_inv ...
    ];

train_loss = [
    train_loss_l_bfgs_2loop, ...
    train_loss_multi_bfgs, ...
    train_loss_l_ms_bfgs_paper, ...
    train_loss_l_ms_bfgs_2loop, ...
    train_loss_l_ms_bfgs_2loop_mu, ...
    train_loss_ms_bfgs_schur_inv ...
    ]/size(y_est, 1); % Normalized by the number of samples

% Misclassification rate
test_misclassify_test = get_misclassify(bt, yt_est);
test_misclassify_l_bfgs_2loop = get_misclassify(bt, At * x_l_bfgs_2loop);
test_misclassify_multi_bfgs = get_misclassify(bt, At * x_multi_bfgs);
test_misclassify_l_ms_bfgs_paper = get_misclassify(bt, At * x_l_ms_bfgs_paper);
test_misclassify_l_ms_bfgs_2loop = get_misclassify(bt, At * x_l_ms_bfgs_2loop);
test_misclassify_l_ms_bfgs_2loop_mu = get_misclassify(bt, At * x_l_ms_bfgs_2loop_mu);
test_misclassify_ms_bfgs_schur_inv = get_misclassify(bt, At * x_ms_bfgs_schur_inv);

test_misclassify = [
    test_misclassify_test, ...
    test_misclassify_l_bfgs_2loop, ...
    test_misclassify_multi_bfgs, ...
    test_misclassify_l_ms_bfgs_paper, ...
    test_misclassify_l_ms_bfgs_2loop, ...
    test_misclassify_l_ms_bfgs_2loop_mu, ...
    test_misclassify_ms_bfgs_schur_inv ...
    ];

%% -------- graph : f-f* -------- %%
graph = trajectory_loss;
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
title('Misclassification Loss (log-log)', FontSize=26)

% Save the figure as a PNG file
saveas(gcf, fullfile('fig' ,'mnist_figure.png'))

% save the workspace
save(fullfile('fig', 'mnist_workspace.mat'));

