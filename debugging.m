%  Verify Brute-Force version of the
%  Limited Memory MS BFGS version

clc, clf, clear; warning('off');
[m, n, eig_range, class_balance, logreg_eps, stepsize, p, sigma, num_iter, iter_limit, seed, signal] = get_parameter();
rng(seed);

[fn, grad, prob_difficulty, matrix_A, y_sol] = ...
logistic_regression(m, n, seed, sigma, class_balance, logreg_eps, eig_range, signal);

B = eye(n); 
x0 = zeros(n,1); 
f0 = fn(x0);

[f_l_ms_bfgs_ours_2loop, traj_l_ms_bfgs_ours_2loop, x_l_ms_bfgs_ours_2loop] = ...
    l_ms_bfgs_ours_2loop(x0, stepsize, num_iter, p, fn, grad);

[f_brute_force, traj_brute_force, x_brute_forcep] = ...
    l_ms_bfgs_brute_force(x0, stepsize, num_iter, p, fn, grad);


trajectory = [traj_l_ms_bfgs_ours_2loop, traj_brute_force];

graph = trajectory_bfgs - f_optimal;
loglog(graph,'-O', 'MarkerSize', 3)
legend({'L-MS-BFGS (two loop)', ...
    'L-MS-BFGS (brute force)' , ...
    }, Location="southwest", Fontsize=14)

xlabel("Iteration", FontSize=30)
ylabel("f(x)", 'Rotation', 0, Fontsize=30)
title('Logistic Regression Loss (log-log plot)', FontSize=26)