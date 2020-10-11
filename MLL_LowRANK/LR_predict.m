function [pre_label, res_once] = LR_predict(W, test_data, test_target)
    pre_value = test_data * W ;
    pre_label = pre_value;
    pre_label(pre_label>0.5) = 1;
    pre_label(pre_label<=0.5) = 0;
    cd('measures');
        HammingLoss = Hamming_loss(pre_label', test_target');
        RankingLoss = Ranking_loss(pre_value', test_target');
        OneError = One_error(pre_value', test_target');
        Coverage = coverage(pre_value', test_target');
        Average_Precision = Average_precision(pre_value', test_target');
       
    cd('..');
    res_once = [HammingLoss, RankingLoss, OneError, Coverage, Average_Precision];
end

