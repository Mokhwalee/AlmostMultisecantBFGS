% manual debugging
% just ramdomly make a matrix and try to debug it
n = 100;
m = 5;
max_iter = 100;
diff = zeros(max_iter,1);

for iter = 1:max_iter  %iteration

    %% ------- initialization ------- %%
    % ramdomly make Sk and Yk
    Sk = cell(1, m);
    Yk = cell(1, m); 
    for j = 1:m
        Sk{j} = rand(n, m);
        Yk{j} = rand(n, m);
    end

    % define the gradient 
    gk = rand(n, 1);

    % define the gamma
    s_km1 = Sk{end}(:, end); y_km1 = Yk{end}(:, end);
    gamma = (y_km1(:,end)'*s_km1(:,end))/(y_km1(:,end)'*y_km1(:,end));


    %% --------- L-MS-BFGS-2loop recursion --------- %%
    q = gk;
    Winv = cell(m,1);
    for i=1:m
        L = size(Yk{i},2);
        Winv{i} = (Yk{i}'*Sk{i})\eye(L);
    end
    for j = m:-1:1
        alpha{j} = Winv{j}'*Sk{j}'*q;
        q = q - Yk{j}*alpha{j};
    end
    r = gamma*q;
    for j = 1:m
        beta = Winv{j}*(Yk{j}'*r); % scalar
        r = r + ( Sk{j} * (alpha{j}-beta) );
    end
    Hkgk_2loop = r + 0;
    

    %% ---------- L-MS-BFGS-Brute_Force ---------- %%
    H = gamma*eye(n); % initialization of the H0
    for i=1:m
        L = size(Yk{i},2);
        W = (Yk{i}'*Sk{i})\eye(L);
        V = eye(n) - Yk{i}*W'*Sk{i}';
        H = V'*H*V + Sk{i}*W'*Sk{i}';
    end
    Hkgk_brute_force = H*gk;

    diff(iter) = norm(Hkgk_2loop - Hkgk_brute_force);
    diff_normalized = (diff - min(diff)) / (max(diff) - min(diff));
end

% Plot the normalized diff array
figure;
plot(diff_normalized);
title('Normalized Difference between L-MS-BFGS-2loop and L-MS-BFGS-Brute_Force');
xlabel('Iteration');
ylabel('Normalized Difference');