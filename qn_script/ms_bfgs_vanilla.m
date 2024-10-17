function [f_optimal, traj_opt, x_opt, traj_grad, error_rate] = ...
    ms_bfgs_vanilla(B, x0, stepsize, max_iter, p, fn, grad, x_sol)

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
    
            if mod(iter,50)==0, B = eye(size(x,1)); end
    
            one = ymem*((ymem'*smem)\ymem');
            two = B*smem*((smem'*B*smem)\(smem'*B));
            B = B + one - two;
            xn = x - (B\grad(x))*stepsize;
        end
    
        s = xn - x;    
        y = grad(xn)-grad(x);   
        smem = [smem,s];    ymem = [ymem,y];
    
        % simplest multisecant
        while size(smem,2) > p 
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