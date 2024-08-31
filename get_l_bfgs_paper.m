function Hkgk = get_l_bfgs_paper(Sk, Yk, gk) 

    % "Single Secant" l-bfgs
    % page 11, Algorithm 3.1
    
    m = size(Sk,2);
    
    Sg_k = Sk'*gk;
    Yg_k = Yk'*gk;
    YY_k = Yk'*Yk;
    
    Rk = eye(m);
    Dk = eye(m);
    for i=1:m % for i<=j
        for j=i:m
            sy = Sk(:,i)'*Yk(:,j);
            Rk(i,j) = sy;
            if i==j, Dk(i,i) = sy; end 
        end
    end
    
    % gamma
    s_km1 = Sk(:, end); 
    y_km1 = Yk(:, end);
    rk = (y_km1'*s_km1) / (y_km1'*y_km1);
    
    % Compute p 
    % inv(R) seems more stable than backslash
    % for stepsize 0.1, if backslash worked better but not stable for 0.01
    %p1 = (Rk')\(Dk+rk*YY_k) * (Rk\Sg_k) - rk*(Rk')\Yg_k;
    p1 = inv(Rk')*(Dk+rk*YY_k)*inv(Rk)*Sg_k - rk*inv(Rk')*Yg_k;
    p2 = -Rk\Sg_k;
    p = [p1; p2];
    
    % Compute Hk*gk
    Hkgk = rk*gk + [Sk, rk*Yk]*p;
    
end
    