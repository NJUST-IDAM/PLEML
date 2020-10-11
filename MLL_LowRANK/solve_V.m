function [V] = solve_V(W,V_t,Z,X,lam3)
    V = V_t+lam3*(X*W-Z);   
end