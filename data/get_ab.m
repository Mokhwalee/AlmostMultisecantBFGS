function [X, y] = get_ab(X, y, a, b)
    idx = (y == a) | (y == b);
    X = normalize_X(X(idx, :));
    y = y(idx);
    y(y == a) = -1;
    y(y == b) = 1;
end