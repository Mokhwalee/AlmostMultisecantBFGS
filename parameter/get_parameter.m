function [m, n, eig_range, class_balance, logreg_eps, stepsize, p, sigma, ...
          num_iter, iter_limit, seed, signal] = get_parameter()

    % Problem size m by n
    m = 500;
    n = 200;
    
    class_balance = 0.5;
    logreg_eps = 0.0001;
    stepsize = 0.01;
    num_iter = 100;
    iter_limit = 10;

    p = 5; % p=L
    seed = 1;

    %{  
    'signal' is used to create c_bar where
    c_bar = exp(-linspace(0, eig_range, n))'
    c = randn(n,1).*(1-c_bar); 
    %}
    signal = 0; 
    eig_range = 20; % 10, 20, 30
   
    % adjust the error of the data
    % if sigma is too big, a is dominated by the W where
    % W = sigma*randn(m,n).*(ones(m,1)*c_bar');
    % if sigma = 1, clear signal, if sigma=10, not clear signal/too much
    % signal
    sigma = 5; % 1, 10, 30

end