function [info_output] = error_info(predicted_labels, actual_labels)
    assert(size(predicted_labels,1) == 1,'Ther is not a vector');
    assert(size(predicted_labels,2) == size(actual_labels,2),'Ther are not the same size');   
    tp = 0; %True positive
    tn = 0; %True negative
    fp = 0; %False positive
    fn = 0; %False negative
    
    for i = 1:size(predicted_labels,2)
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
    info_output.error = numel(info_output.bad) / size(predicted_labels, 2);
    info_output.mcc = (tp*tn-fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); %Matthews correlation coefficient
end