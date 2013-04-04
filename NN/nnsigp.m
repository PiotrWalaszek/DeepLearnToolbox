function [err, bad] = nnsigp(nn, x, y)
%NNSIGP Calculate erros for signalP network
% For a network with 4 output classes in the follwing order the error
% measure in [...] is calculated and returned:
%       1) signalpeptide (S)    [MCC]
%       2) cleavage site (C)    [Specificity , precision, MCC]
%       3) transmembrane (T)    [MCC]
%       4) other         (.)    No error calculated
% The errors are returned in the following order
%       1) signalpeptide MCC
%       2) Cleavage site specificity
%       3) Cleavage site precision
%       4) Cleacage site MCC
%       5) transmembrane MCC

bad = [];
n_samples = size(x,1);
n_output = size(y,2);

assert(n_output~=1,'Behavior of matthew correlation not tested with single output')

% predict labels with network

pred = nnpredict(nn, x);


% find correct targets
[~, expected] = max(y,[],2);

if nn.isGPU
    confusionmat = gpuArray.zeros(2,2,n_output);
else
    confusionmat = zeros(2,2,n_output);
end

if nn.isGPU
    err = gpuArray.zeros(1,5);
else
    err = zeros(1,5);
end


for target_class = 1:n_output    % testing: set to four
    
    %create binary vectors for each class. For each class (target_class)
    % match the predition with target class and the expected class with the
    % target class
    pred_class = ~(pred     == target_class);
    true_class = ~(expected == target_class);
    
    
    %                preduction
    %  ____________________________
    %             |   pos     neg
    %   __________|_________________ 
    %   true | pos| 1,1 TP | 1,2 FN
    %   clas | neg| 2,1 FP | 2,2 TN
    [TP,TN,FP,FN ] =  calcconfusion(pred_class,true_class);
    confusionmat(:,:,target_class) = [TP FN; FP TN];
    
    
end

% fill out errors 
err(1) = matthew(confusionmat(:,:,1));       % 1) signalpeptide(1) MCC
err(2) = specificity(confusionmat(:,:,2));   % 2) Cleavage site(2) specificity
err(3) = precision(confusionmat(:,:,2));   % 3) Cleavage site(2) precision
err(4) = matthew(confusionmat(:,:,2));   % 4) Cleavage site(2) MCC
err(5) = matthew(confusionmat(:,:,3));     % 5) transmembrane(3) MCC


    function [TP,TN,FP,FN ] =  calcconfusion(pred_class,true_class)
        positives = 0;  % definition for readability
        negatives = 1;  % definition for readability
        % http://en.wikipedia.org/wiki/Confusion_matrix
        % read this as First parentesis: ~=  -> false, == -> true 
        % second parentesis: positives / negatives
        % example (~=)   and (...== negatives) = false negatives 
        TP = sum( (pred_class == true_class) .* (true_class == positives) ); %True positive
        TN = sum( (pred_class == true_class) .* (true_class == negatives) ); %True negative
        FN = sum( (pred_class ~= true_class) .* (pred_class == negatives) ); %False negatives
        FP = sum( (pred_class ~= true_class) .* (pred_class == positives) ); %False positives
        
    end

    function prec = precision(confusion)
        % http://en.wikipedia.org/wiki/Accuracy_and_precision
        tp   = confusion(1,1);
        fp   = confusion(2,1);
        prec = tp / (tp + fp);
        if (tp + fp) ==0
            prec = 0;
        end
    end

    function spec = specificity(confusion)
        % http://en.wikipedia.org/wiki/Sensitivity_and_specificity
        fp   = confusion(2,1);
        tn   = confusion(2,2);
        spec = tn / (fp + tn);
        if (fp+tn) ==0
            spec = 0;
        end        
    end

    function mcc =  matthew(confusion)
        % http://en.wikipedia.org/wiki/Matthews_correlation_coefficient  
        % claculates matthew correlation
        tp = confusion(1,1);
        fn = confusion(1,2);
        fp = confusion(2,1);
        tn = confusion(2,2);
              
        %check if mcc denominator is belew zero, set to 1 if so
        mcc_denom = (tp+fp) * (tp+fn) * (tn+fp) * (tn + fn);
        if mcc_denom == 0
            mcc_denom = 1;
        end
        
        mcc = (tp * tn - fp * fn) ./ sqrt(mcc_denom);
        
        % set mcc to zero if any entries in the conf mat is below 5
        if any(confusion(:) < 5) || sum(confusion(:)) < 20
            mcc = 0;   %MCC is ill defined for small numbers ???
        end
    end

end


