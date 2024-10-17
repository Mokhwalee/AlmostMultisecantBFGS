function [f_optimal, traj_opt, x_opt, traj_grad] = single_bfgs_vanilla(B, x0, stepsize, max_iter, fn, grad)

% single secant BFGS
traj_opt = Inf(max_iter,1);
traj_grad = Inf(max_iter,1);
x = x0;

x_opt = x0;
f_optimal = Inf;

for iter = 1:max_iter
    traj_grad(iter) = norm(grad(x))^2;
    if iter == 1
        xn = x - grad(x)*stepsize;
    else
        B = B + (y)*(y')/(y'*s) - ((B*s)*(B*s)')/(s'*B*s);
        xn = x - (B\grad(x))*stepsize;
    end
    
    s = xn - x;
    y = grad(xn)-grad(x);
    
    x = xn;
    traj_opt(iter) = fn(x);
    
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