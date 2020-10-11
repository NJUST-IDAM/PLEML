function [target,gradient] = solve_W_lbfgs(X,Y,W_t,Z,V, lam1,lam3)
    D =X*W_t;
    E_1 = D - Y;
    E_2 = norm(W_t, 'fro').^2;
    E_3 = sum(sum(V.*(D-Z)));
    E_4 = lam3/2*norm((D-Z),'fro').^2;
    target = trace(E_1'*E_1)/2 + lam1*E_2 + E_3 + E_4;
    gradient = X'*E_1+2*lam1*W_t+X'*V+ lam3*X'*(D-Z);
end
