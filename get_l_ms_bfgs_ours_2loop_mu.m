% Description: This function computes the L-BFGS update for the Hessian-vector product
% Return H(k)*g(k) Limited Multisecant(MS) without additional mu*I term

function Hkgk = get_l_ms_bfgs_ours_2loop_mu(Sk, Yk, gk) % multisecant size

    % Sk and Yk are cell arrays
    q = gk;
    m = size(Sk,2);
    alpha = cell(m,1);

    %inv(W) in MS version (rho in single secant version)
    Winv = cell(m,1);
    for i=1:m
        L = size(Yk{i},2);
        %Winv{i} = inv(Yk{i}'*Sk{i}); 
        Winv{i} = (Yk{i}'*Sk{i})\eye(L);
    end

    % two-loop recursion
    for j = m:-1:1
        alpha{j} = Winv{j}'*(Sk{j}'*q);
        q = q - Yk{j}*alpha{j};
    end

    s_km1 = Sk{end}(:, end); y_km1 = Yk{end}(:, end);
    gamma = (y_km1(:,end)'*s_km1(:,end))/(y_km1(:,end)'*y_km1(:,end));
    r = gamma*q;

    for j = 1:m
        beta = Winv{j}*(Yk{j}'*r); % scalar
        r = r + ( Sk{j} * (alpha{j}-beta) );
    end
    
    % get mu
    iter_limit = 10;
    W = [Winv{end}, eye(size(Yk{end},2)); 
         eye(size(Yk{end},2)), gamma*Sk{end}'*Sk{end}];
    D1 = [Yk{end}, gamma*Sk{end}];
    D2 = D1 + 0.;
    mu = getmu(W, D1, D2, iter_limit);

    Hkgk = r + mu*gk;

end
    