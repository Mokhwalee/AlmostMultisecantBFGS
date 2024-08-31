function [fn, grad, hessian] = quadratic(m,n)

    % input : bA=[A;b] stacked by row
    % (Ax-b)'(Ax-b)=(x'A'-b')(Ax-b) = x'A'Ax - 2x'A'b + b'b : qudratic fcn
    
    A = randn(m,n);
    b = rand(m,1);
    AA = A'*A; Ab = A'*b;
    fn = @(x)(x'*(AA*x-2*Ab)+b'*b)/2.0/m; % (Ax-b)^2 / 2m
    grad = @(x)(AA*x-Ab)/m; % (Ax-b)^2' = 2A'(Ax-b)
    hessian = AA/m;
    
    end