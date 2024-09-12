function [m, n, eig_range, class_balance, logreg_eps, stepsize, p, sigma, num_iter, iter_limit, seed, signal] = get_parameter()
% Problem size m by n
m = 100;
n = 80;
eig_range = 20; % 10, 20, 30
class_balance = 0.5;
logreg_eps = 0.0001;
stepsize = 0.01;
p = 6; % p=L
sigma = 1; % 1, 10, 30
num_iter = 20;
iter_limit = 10;
seed = 6; % random integer generator
signal = 1; % 1: high signal, 0: low signal
end