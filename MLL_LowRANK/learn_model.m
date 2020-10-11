function [W, Z, V] = learn_model(X, Y,params)
   
    lam1 = params.lambda1;
    lam2 = params.lambda2;
    min_loss_margin = params.minimumLossMargin;
    max_iter  = params.maxIter;
    
    rho_max=10.^6;
   
    rho=10.^(-6);
    beta=1.1;
    last_loss = 0;
    iter = 0;
    [num_instance, num_dim] = size(X);
    [num_class] = size(Y, 2);
    W = eye(num_dim, num_class);
    Z = eye(num_instance, num_class);
%     W = (X'*X + eye(num_dim, latent_c)) \ (X'*V);
    V = eye(num_instance, num_class);
    
    while iter < max_iter
        %fprintf('======================= V finished: %s ======================= \n', datestr(now));
        [W] = solev_by_lbfgs(@(W)solve_W_lbfgs(X, Y,W, Z , V,lam1,rho),W);
%         fprintf('======================= iter: %d  %s ======================= \n', iter, datestr(now));
        [Z] = solve_Z(W,V,X,lam2, rho);      
%       fprintf('======================= U finished: %s ======================= \n', datestr(now))
        [V] = solve_V(W, V, Z,X, rho);
       % fprintf('======================= W finished: %s ======================= \n', datestr(now));
        
        loss_1 = X*W - Y;
        loss_2 = norm(W, 'fro' ).^2;
        loss_3 = sum(svd(Z));
        target = trace(loss_1'*loss_1)/2 + lam1*loss_2 + lam2*loss_3;
        
        rho = min([rho*beta, rho_max]);
        if abs(target - last_loss) < min_loss_margin  
            break;
        else
            last_loss = target;
        end
        iter = iter + 1;
    end
end