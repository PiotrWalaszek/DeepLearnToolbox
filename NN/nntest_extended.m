function [info_output] = nntest_extended(nn, x, y)
    assert(size(y,2) == 2,'Ther is not two class classification');     
    %Class 1 is negative and class 2 is positive
    
    predicted_labels = nnpredict(nn, x);
    [~, actual_labels] = max(y,[],2);
    
    tp = 0; %True positive
    tn = 0; %True negative
    fp = 0; %False positive
    fn = 0; %False negative
    
    for i = 1:size(y,1)
        if actual_labels(i) == 1
            if predicted_labels(i) == 1
                tn = tn + 1;
            else
                fp = fp + 1;
            end
        else %actual lables == 2 
            if predicted_labels(i) == 2
                tp = tp + 1;
            else
                fn = fn + 1;
            end
        end
    end
    
    info_output.tp = tp;
    info_output.tn = tn;
    info_output.fp = fp;
    info_output.fn = fn;
    info_output.bad = find(predicted_labels ~= actual_labels);    
    info_output.error = numel(info_output.bad) / size(x, 1);
    info_output.mcc = (tp*tn-fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); %Matthews correlation coefficient
end