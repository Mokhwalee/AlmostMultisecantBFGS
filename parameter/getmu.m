function mu = getmu(W, D1, D2, iter_limit)

    [n,p] = size(D1);   p = p/2;
    mu = .01;
    iter = 1;
    c = 1;
    eps = .3;
    [U,Sig,V] = svd(W);
    Sig = 1./diag(Sig);
    S2 = (1+sqrt(1+4*Sig.^2*c^2))./(2*Sig);
    F = c*eps/(c+norm(W)) * V*diag(S2)*U';

    %{
    P = inv(c*eye(2*p) - (1/c)*F*F');
    Q = inv(c*eye(2*p) - (1/c)*F'*F);
    invW = inv(W);
    %}

    P = (c*eye(2*p) - (1/c)*F*F') \ eye(2*p);
    Q = (c*eye(2*p) - (1/c)*F'*F) \ eye(2*p);
    invW = W\eye(2*p);


    B1 = [P,-c*Q*F'-invW ; -c*F*Q-invW',Q];
    B1 = (B1+B1')/2;
    %invB1 = inv(B1);
    invB1 = B1\eye(4*p);
    DTD = [D1,D2]'*[D1,D2];
    
    while iter<iter_limit 
        H2 = [c*eye(2*p), F; F', c*eye(2*p)] - 1/(2*mu)*DTD + ...
             1/(2*mu)^2* DTD * ((invB1 + DTD/2/mu)\DTD);     
        if min(eig(H2+H2'))>1e-15
            break;    
        else
            mu = mu*2; iter = iter+1;
        end
    end    

end