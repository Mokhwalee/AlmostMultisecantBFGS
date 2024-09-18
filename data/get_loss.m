% Calculate the loss as the sum of squared differences
function loss = get_loss(y, y_est)
    loss = sum((y_est - y).^2);
end