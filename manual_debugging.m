% manual debugging
clc, clf, clear; warning('off')
seed = 1; % random integer generator
rng(seed);

% just ramdomly make a matrix and try to debug it
n = 100;
m = 15;
max_iter = 25;
diff = zeros(max_iter,1);
diff_normalized = zeros(max_iter,1);
diff_old = zeros(max_iter,1);
diff_normalized_old = zeros(max_iter,1);

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
    diff_normalized(iter) = norm(Hkgk_2loop - Hkgk_brute_force)/max(norm(Hkgk_2loop),norm(Hkgk_brute_force));

    
    
    %%
    [Hkgk_2loop, Hkgk_brute_force]

    %% ---------- Old version of Brute Force and Extended ---------- %%
    % Brute Force version
    Hk = gamma*eye(n);
    for i=1:m
        SWS= Sk{i}*((Yk{i}'*Sk{i})'\Sk{i}');
        V = eye(n) - Yk{i}*((Yk{i}'*Sk{i})'\Sk{i}');
        Hk = V'*Hk*V + SWS;
    end
    Hkgk_brute_force_old = Hk*gk;

    % Extended version
    SWS_cell = {}; V_cell = {};
    for i=1:m
        SWS_cell{i} = Sk{i}*((Yk{i}'*Sk{i})'\Sk{i}'); % SW'S'
        V_cell{i} = eye(n) - Yk{i}*((Yk{i}'*Sk{i})'\Sk{i}'); % V = I-YW'S'
    end
    Hk = zeros(n); V_product = eye(n); %Hk = rk*eye(n);
    for i=m:-1:1
        if i==m
            Hk = Hk + SWS_cell{i}; % S(k-1)*W(k-1)'*S(k-1)'
        else
            V_product = V_cell{i+1}*V_product;
            Hk = Hk + V_product'*SWS_cell{i}*V_product;
        end
    end
    V_product = V_cell{1}*V_product;
    Hk = Hk + V_product'*gamma*V_product; 
    Hkgk_extended_old = Hk*gk;

    diff_old(iter) = norm(Hkgk_brute_force_old - Hkgk_extended_old);
    diff_normalized_old = (diff - min(diff)) / (max(diff) - min(diff));

end


% Create a figure with two subplots
figure;

% Plot the normalized diff array in the first subplot
subplot(2, 1, 1);
semilogy(diff_normalized);
title('Normalized Difference between L-MS-BFGS-2loop and L-MS-BFGS-Brute-Force');
xlabel('Iteration');
ylabel('Normalized Difference');
%ylim([0, 1e-5]); % Set the y-axis limits

% Plot the normalized diff_old array in the second subplot
subplot(2, 1, 2);
plot(diff_normalized_old);
title('Normalized Difference (Old) between L-MS-BFGS-2loop and L-MS-BFGS-Brute-Force');
xlabel('Iteration');
ylabel('Normalized Difference (Old)');
ylim([0, 1e-2]); % Set the y-axis limits

