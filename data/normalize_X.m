function X = normalize_X(X)
    X = double(X);
    idx = X ~= 0;
    X = X / 255;
    X(idx) = X(idx) - mean(X(idx));
end