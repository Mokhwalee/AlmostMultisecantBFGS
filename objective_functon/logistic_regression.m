function [fn,grad,prob_difficulty, A, b] = logistic_regression(m,n,seed,sigma,class_balance,logreg_eps,eig_range,signal)

if ~isnan(seed)
    rng(seed);
end

if isnan(class_balance)
    class_balance = 0.5;
end

if isnan(logreg_eps )
    logreg_eps = 0;
end

% version 2 : High Signal Case
c_bar = exp(-linspace(0, eig_range, n))'; % each feature has differnt weight
if signal == 0 % low
    c = randn(n,1); % distance
elseif signal == 1 % high
    c = randn(n,1).*(1-c_bar);  
end

W = sigma*randn(m,n).*(ones(m,1)*c_bar'); % error of the data
b = 2*(rand(m,1) > class_balance)-1;
A = b*c' + W;                             % data in R^{m x d}

bA = (repmat(b,1,n).*A); 
sigmoid = @(x)(1./(1+exp(-x)));

if logreg_eps == 0
    fn = @(x)(mean(-log(sigmoid(bA*x))));
else
    fn = @(x)(mean(-log(max(logreg_eps,sigmoid(bA*x)))));
end

grad = @(x)((bA'*(sigmoid(bA*x)-1))/m);

if signal == 0
    prob_difficulty = 'Low Signal Case';
elseif signal == 1
    prob_difficulty = 'High Signal Case';
end

end
