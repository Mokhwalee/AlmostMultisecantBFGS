function [fn, grad, m, n] = logistic_regression_multi_classification(A, b)
    [m, n] = size(A); % m = length(b)
    bA = (repmat(b,1,n).*A); 
    sigmoid = @(x)(1./(1+exp(-x)));
    fn = @(x)(mean(-log(sigmoid(bA*x))));
    grad = @(x)((bA'*(sigmoid(bA*x)-1))/m);
end

%{
function logistic_regression_multi_classification()
    % Load MNIST dataset
    [X, y] = load_mnist_data();
    
    % Normalize pixel values
    X = double(X) / 255.0;
    
    % Split into training and testing sets
    [X_train, y_train, X_test, y_test] = train_test_split(X, y, 0.8);
    
    % Number of classes
    num_labels = 10;
    
    % Train logistic regression model using one-vs-all strategy
    all_theta = one_vs_all(X_train, y_train, num_labels);
    
    % Predict on test set
    pred = predict_one_vs_all(all_theta, X_test);
    
    % Evaluate the model
    accuracy = mean(double(pred == y_test)) * 100;
    fprintf('Accuracy: %.2f%%\n', accuracy);
end

function [X_train, y_train, X_test, y_test] = train_test_split(X, y, train_ratio)
    % Split the data into training and testing sets
    m = size(X, 1);
    idx = randperm(m);
    train_size = round(train_ratio * m);
    
    X_train = X(idx(1:train_size), :);
    y_train = y(idx(1:train_size));
    X_test = X(idx(train_size+1:end), :);
    y_test = y(idx(train_size+1:end));
end

function [X, y] = load_mnist_data()
    % Load the MNIST dataset
    % This function should be implemented to load the MNIST data
    % For example, using the 'loadMNISTImages' and 'loadMNISTLabels' functions
    % from the MNIST helper functions available online.
    X = loadMNISTImages('train-images-idx3-ubyte');
    y = loadMNISTLabels('train-labels-idx1-ubyte');
end

function all_theta = one_vs_all(X, y, num_labels)
    % Train multiple logistic regression classifiers using one-vs-all strategy
    m = size(X, 1);
    n = size(X, 2);
    
    % Add intercept term to X
    X = [ones(m, 1) X];
    
    % Initialize parameters
    initial_theta = zeros(n + 1, 1);
    lambda = 1; % Regularization parameter
    options = optimset('GradObj', 'on', 'MaxIter', 50);
    
    all_theta = zeros(num_labels, n + 1);
    
    for c = 1:num_labels
        % Train logistic regression for class c
        [theta] = fminunc(@(t)(cost_function_reg(t, X, (y == (c-1)), lambda)), initial_theta, options);
        all_theta(c, :) = theta';
    end
end

function [J, grad] = cost_function_reg(theta, X, y, lambda)
    % Compute cost and gradient for logistic regression with regularization
    m = length(y);
    
    h = sigmoid(X * theta);
    J = (1/m) * sum(-y .* log(h) - (1 - y) .* log(1 - h)) + (lambda/(2*m)) * sum(theta(2:end).^2);
    
    grad = (1/m) * (X' * (h - y));
    grad(2:end) = grad(2:end) + (lambda/m) * theta(2:end);
end

function g = sigmoid(z)
    % Compute sigmoid function
    g = 1 ./ (1 + exp(-z));
end

function p = predict_one_vs_all(all_theta, X)
    % Predict the class for each example in X using the trained one-vs-all classifiers
    m = size(X, 1);
    X = [ones(m, 1) X];
    
    % Compute the probability for each class
    probs = sigmoid(X * all_theta');
    
    % Predict the class with the highest probability
    [~, p] = max(probs, [], 2);
    p = p - 1; % Adjust for zero-indexed labels
end

%}