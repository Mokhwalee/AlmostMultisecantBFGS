% Description: This function computes the L-BFGS update for the Hessian-vector product
% Return H(k)*g(k) Limited Multisecant(MS) without additional mu*I term

function Hkgk = get_l_ms_bfgs_2loop_mu(Sk, Yk, gk, iter_limit) % multisecant size

    % Sk and Yk are cell arrays
    q = gk;
    m = size(Sk,2);
    alpha = cell(m,1);

    Winv = cell(m,1);
    for i=1:m
        L = size(Yk{i},2);
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
   
    %{
    % get mu and set up Bk=gamma*I
    W = [-Yk{end}'*Sk{end}, eye(size(Yk{end},2)); 
         eye(size(Yk{end},2)), gamma*Sk{end}'*Sk{end}];
    D1 = [Yk{end}, gamma*Sk{end}];
    D2 = D1 + 0.;
    mu = getmu(W, D1, D2, iter_limit);
    %}

    % Copy and Paste from IEEE version ("get_ms_bfgs_schur_inv_mu.m")
    rk = (y_km1'*s_km1) / (y_km1'*y_km1);
    Wk_inv = Yk{end}'*Sk{end}; 
    YY_k = Yk{end}'*Yk{end};
    one = rk*YY_k + Wk_inv;
    two = Wk_inv;
    W = [one, two; two', zeros(m)];
    D = [rk*Yk{end}, Sk{end}];
    mu = getmu(W, D, D, iter_limit);     


    Hkgk = r + mu*gk;

end
    