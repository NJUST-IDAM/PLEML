function [Z] = solve_Z(W,V,X,lam2, rho)
    D=X*W;
    G=V./rho;
    [U,S,VT]=svd(D+G);
    [c,r]=size(S);
    S_new = zeros(c,r);
    for i=1:c
        for j=1:r
            a = S(i,j)-lam2/rho;
            if a>0 
               S_new(i,j)=a;
            else
               S_new(i,j)=0;
            end
       end
    end
    temp=diag(diag(S_new));
    [c1,~]=size(temp);
    [height,width] = size(D);
    if length(temp)<width
       temp = [temp, zeros(c1,width-c1)];
    elseif length(temp)<height
       temp = [temp ; zeros(height-c1,width)];
    end
    Z=U*temp*VT;
    
end