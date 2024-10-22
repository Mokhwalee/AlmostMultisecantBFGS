function [m, n, eig_range, class_balance, logreg_eps, stepsize, p, sigma, ...
          num_iter, iter_limit, seed, signal] = get_parameter()

    %{  
    m by n : problem size 

    'signal' is used to create c_bar where
        c_bar = exp(-linspace(0, eig_range, n))'
        c = randn(n,1).*(1-c_bar); 
    
    'sigma' adjust the error of the data
    If sigma is too big, a is dominated by the W where
        W = sigma*randn(m,n).*(ones(m,1)*c_bar');
        if sigma = 1, clear signal, if sigma=10, not clear signal/too much
    %}

    m = 200;  % #data
    n = 100; % #feature
    
    class_balance = 0.5;
    logreg_eps = 1e-15;
    stepsize = 0.1;
    num_iter = 1000;
    iter_limit = 10; % max iter for mu*=2*mu

    p = 7; % p=L
    seed = 3;
    signal = 1; 
    eig_range = 20; % 10, 20, 30
    sigma = 1; % 1, 10, 30 # bigger sigma --> eigvals are well distributed

end