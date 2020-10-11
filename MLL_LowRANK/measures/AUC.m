function auc = AUC(outputs, targets)
    % outputs: The outputs of model on test data;
    % targets: The ground true of test data N x L; 
    % AUC: This is the fraction that a positive instance is ranked higher than a negative instance
%     clear;
%     clc;
%     load Arts_expres;
%     outputs = exp_pre_distributions{1}';
%     targets = exp_true_labels{1};
    num_class = size(targets, 2);
    auc = 0;
    for i = 1:num_class
       wrong_rank = 0;
       pos_instacnes_output = outputs(targets(:, i)==1, i);
       neg_instances_output = outputs(targets(:, i)==0, i);
       num_pos = size(pos_instacnes_output, 1);
       num_neg = size(neg_instances_output, 1);
       if num_pos == 0 || num_neg == 0
           auc = auc + 1;
       else
           for j =1:num_pos
               wrong_rank = wrong_rank + sum(pos_instacnes_output(j, 1) > neg_instances_output);
           end
           auc = auc + wrong_rank/(num_pos * num_neg);
       end
    end
    auc = auc / num_class;
end