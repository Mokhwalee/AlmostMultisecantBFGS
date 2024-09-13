function Hkgk = get_l_ms_bfgs_paper(Sk, Yk, gk) % multisecant size

    % Paper version of L-MS-BFGS    
    % "Representations of quasi-Newton matrices and their use in limited memory methods"
    % Algorithm 3.1 in page 11

    [n,m] = size(Sk);
    s_km1 = Sk(:, end); 
    y_km1 = Yk(:, end);
    
    % Compute rk, for example,
    rk = (y_km1'*s_km1) / (y_km1'*y_km1);
    
    Sg_k = Sk'*gk;
    Yg_k = Yk'*gk;
    
    Wk = Yk'*Sk; 
    Wk_inv = Wk\eye(m); %inv(Wk); % both works well
    
    % multiplying with gk is better
    SW_t = Sk*Wk_inv';
    SW = Sk*Wk_inv;
    YW_t = Yk*Wk_inv';
    
    one = SW_t*Sg_k;
    two = rk*SW*(Yk'*YW_t*Sg_k);
    three = -rk*YW_t*Sg_k;
    four = -rk*SW*Yg_k;
    
    % Compute Hkgk
    Hkgk = rk*gk + one+two+three+four;
    
end 