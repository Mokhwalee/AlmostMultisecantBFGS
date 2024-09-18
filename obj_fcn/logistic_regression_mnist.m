function [fn, grad, m, n] = logistic_regression_mnist(A, b)
    [m, n] = size(A);
    bA = (repmat(b,1,n).*A); 
    sigmoid = @(x)(1./(1+exp(-x)));
    fn = @(x)(mean(-log(sigmoid(bA*x))));
    grad = @(x)((bA'*(sigmoid(bA*x)-1))/m);
end