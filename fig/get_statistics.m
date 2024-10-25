
function [traj_cell, traj_grad_cell] = get_statistics(trajectories, trajectory_grad_square)

    threshold = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, ...
                 1e-11, 1e-12, 1e-13, 1e-14, 1e-15];

    % Methods : L-BFGS, MS-BFGS, L-MS-BFGS(paper, 2-loop), 
    %           L-MS-BFGS-Schur-inv-mu, MS-BFGS-Schur-inv-mu
    
    traj_cell = {};
    traj_grad_cell = {};

    for method = 1:size(trajectories,2) % method
        traj = [];
        traj_grad = [];
        for i = 1:length(threshold) % threshold
            iter1 = min(find(trajectories(:,method) < threshold(i), 1));
            iter2 = min(find(trajectory_grad_square(:,method) < threshold(i), 1));
            if isempty(iter1)
                iter1 = size(trajectories,1);
            end
            if isempty(iter2)
                iter2 = size(trajectories,1);
            end

            traj = [traj, iter1];
            traj_grad = [traj_grad, iter2];

        end
        traj_cell{method} = traj;
        traj_grad_cell{method} = traj_grad;
    end


end
