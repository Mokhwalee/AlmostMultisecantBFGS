function [f_optimal, traj_opt, x_opt, traj_grad, error_rate] = ...
    l_ms_bfgs_paper(x0, stepsize, max_iter, L, fn, grad, x_sol)

    % Limited Memory multi secant BFGS for H
    
    %n = size(B,1);
    traj_opt = Inf(max_iter,1);
    traj_grad = Inf(max_iter,1);
    error_rate = Inf(max_iter,1);
    smem = []; ymem = [];
    x = x0;

    x_opt = x0; 
    f_optimal = Inf;

    for iter = 1:max_iter
        traj_grad(iter) = norm(grad(x))^2;
        if iter == 1
            xn = x - grad(x)*stepsize;
        else
        
        % Algorithm 3.1 - Compute Hg with limited mememory version
        Bg = get_l_ms_bfgs_paper(smem, ymem, grad(x));
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
    
        % stopping criteria
        if norm(grad(x))<1e-14 || fn(x)<1e-14
            traj_opt(iter+1:max_iter)=fn(x); 
            traj_grad(iter+1:max_iter) = norm(grad(x))^2;
            break; 
        end
    
    end

end