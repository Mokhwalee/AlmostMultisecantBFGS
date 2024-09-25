function [] = run_qn_sensing_problem(n, fn, grad, stepsize, num_iter, p, iter_limit, name, folder_name)
    
    % Initialization
    B = eye(n); 
    x0 = zeros(n,1); 
    
    % Run QN
    [f_optimal, traj_opt, x_opt] = ...
        single_bfgs_vanilla(B, x0, 0.01, 10000, fn, grad);
    
    [f_l_bfgs_2loop, traj_l_bfgs_2loop, x_l_bfgs_2loop] = ...
        l_bfgs_2loop(x0, stepsize, num_iter, p, fn, grad);
    
    [f_multi_bfgs, traj_multi_bfgs, x_multi_bfgs] = ... % Vanilla
        ms_bfgs_vanilla(B, x0, stepsize, num_iter, p, fn, grad);
    
    [f_l_ms_bfgs_paper, traj_l_ms_bfgs_paper, x_l_ms_bfgs_paper] = ...
        l_ms_bfgs_paper(x0, stepsize, num_iter, p, fn, grad); 
    
    [f_l_ms_bfgs_2loop, traj_l_ms_bfgs_2loop, x_l_ms_bfgs_2loop] = ...    
        l_ms_bfgs_2loop(x0, stepsize, num_iter, p, fn, grad);
    
    [f_l_ms_bfgs_2loop_mu, traj_l_ms_bfgs_2loop_mu, x_l_ms_bfgs_2loop_mu] = ...    
        l_ms_bfgs_2loop_mu(x0, stepsize, num_iter, p, fn, grad, iter_limit);
    
    [f_ms_bfgs_schur_inv, traj_ms_bfgs_schur_inv, x_ms_bfgs_schur_inv] = ...
        ms_bfgs_schur_inv_mu(x0, stepsize, num_iter, p, fn, grad, iter_limit);
    
    trajectory_bfgs = [
        traj_l_bfgs_2loop, ...
        traj_multi_bfgs, ...
        traj_l_ms_bfgs_paper, ...
        traj_l_ms_bfgs_2loop, ...
        traj_l_ms_bfgs_2loop_mu, ...
        traj_ms_bfgs_schur_inv  
        ];
    
    graph = trajectory_bfgs - f_optimal;
    loglog(graph,'-O', 'MarkerSize', 3)
    legend({'L-BFGS (baseline, two-loop)', ...
        'MS BFGS (baseline)', ...
        'L-MS-BFGS (paper)', ...
        'L-MS-BFGS (2-loop)', ...
        'L-MS-BFGS-mu (ours)', ...
        'MS-BFGS-Schur-inv-mu (IEEE)', ...
        }, Location="southwest", Fontsize=14)
    
    % Set the font size of the tick labels
    ax = gca;
    ax.XAxis.FontSize = 20; % Change the font size of the x-axis tick labels
    ax.YAxis.FontSize = 20; % Change the font size of the y-axis tick labels
    
    xlabel("Iteration", FontSize=30)
    ylabel("f(x)", 'Rotation', 0, Fontsize=30)
    title('Sensing Problem - Logistic Regression(log-log)', FontSize=26)
    
    % Save the figure as a PNG file
    save(fullfile(folder_name+'/workspace', name+'_workspace.mat')); % workspace
    saveas(gcf, fullfile(folder_name, name+'.png'));    % plot
    


end