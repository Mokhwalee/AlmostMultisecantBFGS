% Brute Force version : Limited Memory single secant BFGS
function [f_optimal, traj_opt, x_opt] = l_bfgs_2loop(x0, stepsize, max_iter, L, fn, grad)

    %n = size(B,1);
    traj_opt = Inf(max_iter,1);
    smem = []; ymem = [];
    x = x0;

    x_opt = x0; 
    f_optimal = Inf;

    for iter = 1:max_iter

        if iter == 1
            xn = x - grad(x)*stepsize;
        else
            q = grad(x);
            m = size(smem,2);
            alpha = zeros(m,1);
            rho = 1./(sum(ymem.*smem,1)); % vector

            % two-loop recursion
            for j = m:-1:1
                alpha(j) = rho(j)*(smem(:,j)'*q);
                q = q - alpha(j)*ymem(:,j);
            end

            gamma = (ymem(:,end)'*smem(:,end))/(ymem(:,end)'*ymem(:,end));
            r = gamma*q;

            for j = 1:m
                beta = rho(j)*(ymem(:,j)'*r); % scalar
                r = r + ( smem(:,j) * (alpha(j)-beta) );
            end

            B = r+0;
            xn = x - B*stepsize;        
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

        % stopping criteria
        if fn(x)<1e-14
            traj_opt(iter+1:max_iter)=fn(x); 
            break; 
        end

    end
    
end