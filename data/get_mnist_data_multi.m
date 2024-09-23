% multi classification task by logistic regression
% Input: num_classes - number of classes to include in the dataset
% Output: X, y, Xt, yt - training and test datasets

function [X, y, Xt, yt] = get_mnist_data_multi(num_classes)
    % num_classes should be between 0 to 9
    % Load the .mat file
    data = load('mnist.mat');
    
    % Extract and process training data
    X = double(data.trainX);
    y = double(data.trainY(1, :))'; % column vector
    
    % Extract and process test data
    Xt = double(data.testX);
    yt = double(data.testY(1, :))'; % column vector

   % Filter the datasets to include only the specified number of classes
   classes = unique(y);
   if length(classes) < num_classes
       error('Number of classes in the dataset is less than the specified num_classes');
   end

   % Select random num_classes classes
   random_indices = randperm(length(classes), num_classes);
   selected_classes = classes(random_indices);

   % Filter training data
   train_filter = ismember(y, selected_classes);
   X = X(train_filter, :);
   y = y(train_filter);

   % Filter test data
   test_filter = ismember(yt, selected_classes);
   Xt = Xt(test_filter, :);
   yt = yt(test_filter);

end