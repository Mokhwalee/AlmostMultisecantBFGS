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

% Generate Problem
clc, clf, clear; warning('off')

% Problem size m by n
m = 1000;
n = 300;

snr = 0.5 % 1, 0.75, 0.5
eig_range = 20 % 10, 20, 30

class_balance = 0.5;
logreg_eps = 0.0001;
stepsize = 0.1;
p = 10; % p=L
sigma = 1; % 1, 10, 30
num_iter = 500;
iter_limit = 30;
seed = 10 % random integer generator
rng(seed);

% Problems : (1)Quadratic, (2)low signal Logreg, (3)high signal Logreg
%[fn, grad] = quadratic(m,n); f0 = fn(x0);

[fn, grad, prob_difficulty, matrix_A, y_sol] = ...
    high_signal_logreg(m,n,seed,sigma,snr,class_balance,logreg_eps,eig_range);
%f0 = fn(x0); %initial function val

%[fn, grad, prob_difficulty, matrix_A, y_sol] = ...
%    low_signal_logreg(m,n,seed,sigma,snr,class_balance,logreg_eps,eig_range);

B = eye(n); x0 = zeros(n,1); f0 = fn(x0);

% optimal with small step-size
[f_optimal, traj_opt, x_opt] = single_bfgs(B, x0, 0.01, 5000, fn, grad);

%% Multisecant BFGS
[f_multi_bfgs, traj_multi_bfgs, x_multi_bfgs] = ... % Vanilla
    multi_bfgs(B, x0, stepsize, num_iter, p, fn, grad);

%[f_symm_multi_bfgs,  traj_symm_multi_bfgs,  x_symm_multi_bfgs] = ...
%    symm_multi_bfgs(B, x0, stepsize, num_iter, p, fn, grad);

%[f_psd_multi_bfgs,   traj_psd_multi_bfgs,   x_psd_multi_bfgs] = ...
%    psd_multi_bfgs(B, x0, stepsize, num_iter, p, fn, grad);

%[f_schur_multi_bfgs, traj_schur_multi_bfgs, x_schur_multi_bfgs] = ...
%    schur_multi_bfgs(B, x0, stepsize, num_iter, p, fn, grad, iter_limit);

%[w_f_schur_multi_bfgs, w_traj_schur_multi_bfgs, w_x_schur_multi_bfgs] = ...
%    w_schur_multi_bfgs(B, x0, stepsize, num_iter, p, fn, grad, iter_limit);

%% Limited MS
[f_L_ms_bfgs, traj_L_ms_bfgs, x_L_ms_bfgs] = ...
    l_ms_bfgs(B, x0, stepsize, num_iter, p, fn, grad); % L-MS-BFGS (paper)

[w_f_schur_L_ms_bfgs, w_traj_schur_L_ms_bfgs, w_x_schur_L_ms_bfgs] = ...
    w_l_ms_bfgs_schur(B, x0, stepsize, num_iter, p, fn, grad, iter_limit);
% Woodbury Schur (paper)

[f_L_ms_bfgs_B, traj_L_ms_bfgs_B, x_L_ms_bfgs_B] = ... % B,Schur (paper)
    l_ms_bfgs_schur(B, x0, stepsize, num_iter, p, fn, grad, iter_limit);

[f_BF_L_ms_bfgs, traj_BF_L_ms_bfgs, x_BF_L_ms_bfgs] = ... % Brute-Force
    BruteForce_ms_bfgs(B, x0, stepsize, num_iter, p, fn, grad);

[f_our_ms_bfgs, traj_our_ms_bfgs, our_L_ms_bfgs] = ...    % Ours(removed)
    our_l_ms_bfgs(B, x0, stepsize, num_iter, p, fn, grad);

[f_our_ms_bfgsib_vanilla, traj_our_ms_bfgs_vanilla, our_L_ms_bfgs_vanilla] = ...
    our_l_ms_bfgs_vanilla(B, x0, stepsize, num_iter, p, fn, grad);
% Ours(Vanilla)

[f_extended_L_ms_bfgs, traj_extended_L_ms_bfgs, x_extended_L_ms_bfgs] = ...
    extended_ms_bfgs(B, x0, stepsize, num_iter, p, fn, grad); % Extended


%% Trajectories
trajectory_bfgs = [%traj_symm_multi_bfgs, ...
    %traj_psd_multi_bfgs, ...
    %traj_schur_multi_bfgs, ...
    %w_traj_schur_multi_bfgs, ...
    traj_multi_bfgs, ...
    traj_L_ms_bfgs, ...
    w_traj_schur_L_ms_bfgs,  ...
    traj_L_ms_bfgs_B, ...
    traj_BF_L_ms_bfgs, ...
    traj_our_ms_bfgs,...
    traj_our_ms_bfgs_vanilla, ...
    traj_extended_L_ms_bfgs];


%% graph : f-f*
graph = trajectory_bfgs - f_optimal;
loglog(graph,'-O', 'MarkerSize', 3)
legend({%'symm multi BFGS',...
    %'psd multi BFGS',...
    %'schur multi BFGS', ...
    %'W schur multi BFGS(H)' ...
    'multi BFGS', ...
    'L-multi BFGS(H) - paper', ...
    'L-multi schur BFGS(H)', ...
    'L-multi schur BFGS(B)', ...
    'L-MS-BFGS(H) Brute Force ',...
    'L-MS-BFGS(H) Ours',...
    'L-MS-BFGS(H) Ours Vanilla',...
    'L-MS-extended(H)'
    }, Location="southwest", Fontsize=16)

xlabel("Iteration", FontSize=30)
ylabel("f(x)      ",'Rotation',0, Fontsize=30)
title('Logistic Regression Loss (log-log)', FontSize=26)








