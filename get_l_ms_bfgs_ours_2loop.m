% Description: This function computes the L-BFGS update for the Hessian-vector product
% Return H(k)*g(k) Limited Multisecant without additional mu*I term

function Hkgk = get_l_ms_bfgs_ours_2loop(Sk, Yk, gk) % multisecant size
    
    Sbar = []; Ybar = [];
    for i=1:m
        Sbar = [Sbar, Sk{i}];
        Ybar = [Ybar, Yk{i}];
    end

    s_km1 = Sk{end}(:, end); y_km1 = Yk{end}(:, end);
    rk = (y_km1'*s_km1) / (y_km1'*y_km1); % % gamma(k), scalar
    
    q = gk;
    m = size(Sk,2);
    alpha = zeros(m,1);
    rho = 1./(sum(Yk.*Sk,1)); % vector

    % two-loop recursion
    for j = m:-1:1
        alpha(j) = rho(j)*(smem(:,j)'*q);
        q = q - alpha(j)*ymem(:,j);
    end

    gamma = (ymem(:,end)'*smem(:,end))/(ymem(:,end)'*ymem(:,end));
    r = gamma*q;

    for j = 1:m
        beta = rho(j)*(ymem(:,j)'*r); % scalar
        r = r + ( smem(:,j) * (alpha(j)-beta) );
    end

    Hkgk = r + 0;

    %{
    m = size(Sk,2); % because Sk is the cell array
    %n = size(Sk{1},1);
    
    % Algo I : (Optional)Remove redundant cols in Sk and Yk
    %{
    for i=1:m
       if size(Sk{1},2)==m
           if i==m, continue; end 
           Sk{i} = Sk{i}(1:end, 1); 
           Yk{i} = Yk{i}(1:end, 1); 
       
       elseif 1<size(Sk{1},2) && size(Sk{1},2)<m
           if i==1, continue; end
           Sk{i} = Sk{i}(1:end, end); 
           Yk{i} = Yk{i}(1:end, end); 
       
       else % if the column size=1, then just take the last one
           Sk{i} = Sk{i}(1:end, end); 
           Yk{i} = Yk{i}(1:end, end); 
       end
    end
    %}
    
    % Algo II : Compute Sk_bar, Yk_bar
    Sbar = []; Ybar = [];
    for i=1:m
        Sbar = [Sbar, Sk{i}];
        Ybar = [Ybar, Yk{i}];
    end
    
    
    % Algo III
    start_idx = [1]; end_idx = [0];
    for i=1:m
        size_sk = size(Sk{i},2);
        start_idx = [start_idx, start_idx(end)+size_sk];
        end_idx = [end_idx, end_idx(end)+size_sk];
    end
    start_idx = start_idx(1:end-1);
    end_idx = end_idx(2:end); % remove the first 0 elem
    
    
    % Algo IV
    StY = {}; D = [];
    for i=1:m
        StY{i} = []; % initialize the cell element
        for j=i:m
            SitYj = Sk{i}'*Yk{j};
            StY{i} = [StY{i}, SitYj];
            if i==j
                D = blkdiag(D, SitYj);
            end
        end
    end
    
    % Algo V
    R = zeros(end_idx(end));
    for i=1:m
        R(start_idx(i):end_idx(i), start_idx(i):end) = StY{i};
    end
    
    % Algo VI : Compute Hk
    s_km1 = Sk{end}(:, end); y_km1 = Yk{end}(:, end);
    rk = (y_km1'*s_km1) / (y_km1'*y_km1); % % gamma(k), scalar
    
    
    % Compute Hk*gk
    Sg_k = Sbar'*gk;
    Yg_k = Ybar'*gk;
    
    
    one = Sbar*(R'\D)*(R\Sg_k);
    two = rk*Sbar*(R'\Ybar')*Ybar*(R\Sg_k);
    three = -rk*Ybar*(R\Sg_k);
    four = -rk*Sbar*(R'\Yg_k);
    
    % Compute Hkgk
    Hkgk = rk*gk + one+two+three+four;

    %}


end
    