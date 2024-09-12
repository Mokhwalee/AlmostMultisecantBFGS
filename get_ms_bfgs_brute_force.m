function Hk = get_ms_bfgs_brute_force(Sk, Yk) % multisecant size

% Brute-Force version of "Multi Secant" l-bfgs
% H+ = V'HV + SW'S'
% where W = inv(Y'S), V = I-YW'S'

n = size(Sk{1},1);
m = size(Sk,2); % because Sk is the cell array

% Compute W=inv(Y'S) and SW'S'
s_km1 = Sk{end}(:, end); y_km1 = Yk{end}(:, end);
rk = (y_km1'*s_km1) / (y_km1'*y_km1); % gamma(k), scalar
Hk = rk*eye(n);

for i=1:m
    %W = inv(Yk{i}'*Sk{i}); 
    %SWS= Sk{i}*W'*Sk{i}';
    %V = eye(n) - Yk{i}*W'*Sk{i}';
    
    % Instead of inv(YS), backslash is FAR better (maybe less err)
    SWS= Sk{i}*((Yk{i}'*Sk{i})'\Sk{i}');
    V = eye(n) - Yk{i}*((Yk{i}'*Sk{i})'\Sk{i}');
    
    % Compute Hk
    Hk = V'*Hk*V + SWS;
end

end
