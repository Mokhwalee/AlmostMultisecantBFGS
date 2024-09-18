% Calculate the loss as the sum of squared differences
function loss = get_loss(y, y_est)
    loss = sum((sign(y_est) - sign(y)).^2);
end