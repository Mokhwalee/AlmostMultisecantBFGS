function [f_optimal, traj_opt, x_opt] = multi_bfgs(B, x0, stepsize, max_iter, p, fn, grad)

    traj_opt = Inf(max_iter,1);
    smem = []; ymem = [];
    x = x0; 
    
    x_opt = x0; 
    f_optimal = Inf;
    
    for iter = 1:max_iter
    
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
        
        if fn(x) < f_optimal
            x_opt = x; 
            f_optimal = traj_opt(iter);
        end
        
        % stopping criteria
        if fn(x)<1e-14, traj_opt(iter+1:max_iter)=fn(x); break; end
    
    end

end