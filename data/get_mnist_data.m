
function [X, y, Xt, yt] = get_mnist_data(a, b)
    % Choose binary parameter 'a' and 'b' to classify
    % For example, a = 4; b = 9;

    % Load the .mat file
    data = load('mnist.mat');

    % Extract and process training data
    X = double(data.trainX);
    y = double(data.trainY(1, :))'; % column vector
    [X, y] = get_ab(X, y, a, b);

    % Extract and process test data
    Xt = double(data.testX);
    yt = double(data.testY(1, :))'; % column vector

    [Xt, yt] = get_ab(Xt, yt, a, b);

end
