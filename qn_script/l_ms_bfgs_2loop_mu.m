% Description: L-MS-BFGS method with our own implementation

function [f_optimal, traj_opt, x_opt, traj_grad, error_rate] = ...
    l_ms_bfgs_2loop_mu(x0, stepsize, max_iter, L, fn, grad, iter_limit, x_sol)

    %n = size(B,1);
    traj_opt = Inf(max_iter,1);
    traj_grad = Inf(max_iter,1);
    error_rate = Inf(max_iter,1);
    smem = []; ymem = [];
    x = x0;
    x_opt = x0; 
    f_optimal = Inf;

    % initialize the cell arrays for L-MS-BFGS
    Sk = {}; Yk = {};

    for iter = 1:max_iter
        traj_grad(iter) = norm(grad(x))^2;
        if iter == 1
            xn = x - grad(x)*stepsize;
        else
            Bg = get_l_ms_bfgs_2loop_mu(Sk, Yk, grad(x), iter_limit);
            xn = x - Bg*stepsize;        
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
        error_rate(iter) = sum(sign(x_sol) ~= sign(x));

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
        
        % exp(-(bA*x)) is not 1 but almost 0; 1.0e-31*0.147507419481987
        % aa = exp(-(bA*x)) % sigmoid becomes infinite-->grad(x)=0
        % norm(grad(x))^2
        % grad = @(x)((bA'*(sigmoid(bA*x)-1))/m); becomes 0

        % stopping criteria
        if norm(grad(x))<1e-14 || fn(x)<1e-14
            x_opt = x; 
            traj_opt(iter+1:max_iter)=fn(x); 
            traj_grad(iter+1:max_iter) = norm(grad(x))^2;
            break; 
        end

    end

end

