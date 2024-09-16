function [f_optimal, traj_opt, x_opt] = ms_bfgs_brute_force(x0, stepsize, ...
                                        max_iter, L, fn, grad)

% Brute Force : Limited Memory multi secant BFGS for H
% Have to save the 'm' previous Sk(=smem) and Yk(=ymem)
traj_opt = Inf(max_iter,1);
smem = []; ymem = [];
x = x0;

x_opt = x0; 
f_optimal = Inf;

% initialize the cell arrays for L-MS-BFGS
Sk = {}; Yk = {};

for iter = 1:max_iter

    if iter == 1
        xn = x - grad(x)*stepsize;
    else
        % Algorithm 3.1 - Compute Hg with limited mememory version
        B = get_ms_bfgs_brute_force(Sk, Yk);
        xn = x - B*grad(x)*stepsize;        
    end

    s = xn - x;    
    y = grad(xn) - grad(x);   
    smem = [smem,s];    ymem = [ymem,y];

    % simplest multisecant
    while size(smem,2) > L
        smem = smem(:,2:end);   ymem = ymem(:,2:end);
    end   
    
    x = xn;
    traj_opt(iter) = fn(x);
    
    if fn(x) < f_optimal
        x_opt = x; 
        f_optimal = traj_opt(iter);
    end

    while size(smem,2) > L
        smem = smem(:,2:end);   ymem = ymem(:,2:end);
    end   
    
    % Save Sk and Yk
    if isempty(Sk)
        Sk{1} = smem; Yk{1} = ymem;
    else
        Sk{end+1} = smem; Yk{end+1} = ymem;
    end

    while size(Sk,2) > L
        Sk = Sk(:, 2:end);   Yk = Yk(:, 2:end);
    end   
    
    % stopping criteria
    if fn(x)<1e-14, traj_opt(iter+1:max_iter)=fn(x); break; end

end



