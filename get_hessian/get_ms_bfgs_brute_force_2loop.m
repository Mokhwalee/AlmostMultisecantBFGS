% Description: Brute Force version of L-MS-BFGS
% Recursively compute H(k+1)

function Hkgk = get_l_ms_bfgs_brute_force(Sk, Yk, gk) % multisecant size

    % Sk and Yk are cell arrays
    n = size(gk,1);
    m = size(Sk,2); %size(Sk{end})
    
    s_km1 = Sk{end}(:, end); y_km1 = Yk{end}(:, end);
    gamma = (y_km1(:,end)'*s_km1(:,end))/(y_km1(:,end)'*y_km1(:,end));
    %gamma = 1;
    
    % Compute H using previous m number of info in a brute-force way
    H = gamma*eye(n); % initialization of the H0
    for i=1:m
        L = size(Yk{i},2);
        W = (Yk{i}'*Sk{i})\eye(L);
        V = eye(n) - Yk{i}*(W'*Sk{i}');
        H = V'*H*V + Sk{i}*(W'*Sk{i}');
    end

    Hkgk = H*gk;

end