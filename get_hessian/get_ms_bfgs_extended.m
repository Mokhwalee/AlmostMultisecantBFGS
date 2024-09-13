function Hkgk = get_ms_bfgs_extended(Sk, Yk, gk) % multisecant size

% L-MS-BFGS

% Brute-Force version of L-MS-BFGS
% Return H(k)*g(k) Limited Multisecant
% Without additional mu*I term

% Here, Sk and Yk are "cells"
% that have 'm' previous matrices

n = size(Sk{1},1);
m = size(Sk,2); 

% Compute W=inv(Y'S), V = I-YW'S' and SW'S'
SWS_cell = {}; V_cell = {};
for i=1:m
    SWS_cell{i} = Sk{i}*((Yk{i}'*Sk{i})'\Sk{i}'); % SW'S'
    V_cell{i} = eye(n) - Yk{i}*((Yk{i}'*Sk{i})'\Sk{i}'); % V = I-YW'S'
end

% gamma 
s_km1 = Sk{end}(:, end); y_km1 = Yk{end}(:, end);
rk = (y_km1'*s_km1) / (y_km1'*y_km1); % scalar, matrix version?

% Compute Hk
Hk = zeros(n); V_product = eye(n); %Hk = rk*eye(n);
for i=m:-1:1
    if i==m
        Hk = Hk + SWS_cell{i}; % S(k-1)*W(k-1)'*S(k-1)'
    else
        % multiply V recursively
        V_product = V_cell{i+1}*V_product;
        Hk = Hk + V_product'*SWS_cell{i}*V_product;
    end
end

% Add V(k-1)'...V(k-m)'H(0)V(k-m)...V(k-1)
V_product = V_cell{1}*V_product;
Hk = Hk + V_product'*rk*V_product; 

% Compute Hkgk
Hkgk = Hk*gk;


end
