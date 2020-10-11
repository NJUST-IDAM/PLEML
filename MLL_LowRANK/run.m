clear;
clc;
fold = 5;
dataset = {'Yeast_spoem','Yeast_spo5','Yeast_spo','Yeast_heat','Yeast_elu','Yeast_dtt','Yeast_diau','Yeast_cold','Yeast_cdc','Yeast_alpha','SJAFFE','SBU_3DFE'};
params = [2,        1;       % Yeast_spoem
          16,    	1;        % Yeast_spo5
          256,      128;      % Yeast_spo
          0.0625,  2;       % Yeast_heat
          4,        32;       % Yeast_elu
          16,       32;       % Yeast_dtt
          16,       16;       % Yeast_diau
          0.25      16;       %Yeast_cold
          16        16;       %Yeast_cdc
          16        16;       % Yeast_alpha
          0.0625,   16;       % SJAFFE
          0.0625,   8;        % SBU_3DFE
          ];        
for exp=1:12
    dataset_name = dataset{exp}
    cd('datasets');
        eval(['load ', dataset_name]);
    cd('..');
    y_log=ch_log(labels,0.5);
    cd('data');
        eval(['save ', dataset_name, '.mat dataset_name features labels y_log']);        
    cd('..');
    cd('data')
        eval(['load ', dataset_name]);
    cd('..');
    num_instance=size(features,1);
    indices=crossvalind('Kfold',num_instance,fold);
    cd('data_fold')
        eval(['save ',dataset_name, '.mat dataset_name features labels y_log indices']);
    cd('..')
    cd('data_fold')
        eval(['load ', dataset_name]);
    cd('..')
    [n,l] = size(labels);
    last_col = ones(n,1);                  
    features = [features, last_col];
    distribution = zeros(n,l);
    % parameters setting
    opt_params.lambda1 = params(exp,1); % 
    opt_params.lambda2 = params(exp,2); % 
    opt_params.maxIter = 100;
    opt_params.minimumLossMargin = 0.0001;

    pre_value = zeros(n,l);
    distribution = zeros(n,l);
    for rep=1:fold
        fprintf('============================ %d %s ============================ \n',rep, datestr(now));
        test_Idx = find(indices == rep);
        train_Idx = setdiff(find(indices),test_Idx);
        test_data = features(test_Idx,:);
        train_data = features(train_Idx,:);
        test_targets = y_log(test_Idx,:);
        train_targets = y_log(train_Idx,:);
        [W, Z, V] = learn_model(train_data, train_targets, opt_params);
        %fprintf('============================  %d %s ============================ \n', rep, datestr(now));
        % Prediction
        pre_label = test_data * W ;
        pre_value(test_Idx,:)=pre_label;
    end
    distribution = softmax(pre_value')';
    cd('result_mll')
             eval(['save ',dataset_name, '_mll.mat dataset_name distribution']);
    cd('..')
end