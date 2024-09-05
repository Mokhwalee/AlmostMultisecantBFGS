function Hkgk = get_ms_bfgs_schur_inv(Sk, Yk, gk, iter_limit) % multisecant size

% Woodbury + mu*I : L-MS-BFGS
% Return H(k)*g(k) Limited Multisecant
% With additional mu*I term

m = size(Sk,2);
s_km1 = Sk(:, end); 
y_km1 = Yk(:, end);

% Compute rk, for example,
rk = (y_km1'*s_km1) / (y_km1'*y_km1);
Wk_inv = Yk'*Sk; % because Wk=inv(Yk'Sk)
YY_k = Yk'*Yk;

% Create W and D in IEEE version
one = rk*YY_k + Wk_inv;
two = Wk_inv;
W = [one, two; two', zeros(m)];
W_inv = W\eye(2*m);
D = [rk*Yk, Sk]; % We put rk*I = Hk

% Default : mu=0;
% Use Schur Complement version to get mu 
mu = getmu(W, D, D, iter_limit);      

% Compute Hkgk : construct nxn matrix, not limited version
Hkgk = (rk+mu)*gk - D*W_inv*D'*gk; % D is in R^{nxL}


end
