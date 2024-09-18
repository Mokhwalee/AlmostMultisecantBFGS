% Calculate the misclassification rate
function misclassify = get_misclassify(y, y_est)
    misclassify = sum(sign(y_est) ~= y) / (length(y) + 0.0);
end