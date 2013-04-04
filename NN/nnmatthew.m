function [mcc, bad] = nnmatthew(nn, x, y)
%MATTHEW calculate matthew coefficient for all target classes
%   Calculates the matthew correlation coefficient for all target calasses.
%   for a n-class classification problem the function returns a n
%   dimensional row vector.
%   bad is notused, but returned for compability with rest of code.



bad = [];
n_samples = size(x,1);
n_output = size(y,2);

assert(n_output~=1,'Behavior of matthew correlation not tested with single output')

% predict labels with network

pred = nnpredict(nn, x);


% find correct targets
[~, expected] = max(y,[],2);

if nn.isGPU
    mcc = gpuArray(zeros(1,n_output));
else
    mcc = zeros(1,n_output);
end
TPt = 0; TNt = 0; FPt = 0; FNt = 0;
for target_class = 1:n_output    % testing: set to four
    
    %create binary vectors for each class. For each class (target_class)
    % match the predition with target class and the expected class with the
    % target class
    pred_class = ~(pred     == target_class);
    true_class = ~(expected == target_class);
    
    %
    [TP,TN,FP,FN,MCC] =  matthew_calc(pred_class,true_class);
    mcc(target_class) = MCC;
    TPt = TPt + TP;
    TNt = TNt + TN;
    FPt = FPt + FP;
    FNt = FNt + FN;
    
    mcc(n_output+1) = (TP * TN - FP * FN) ./ sqrt((TP+FP) * (TP+FN) * (TN+FP) * (TN + FN));
end


    function [TP,TN,FP,FN,MCC] =  matthew_calc(pred_class,true_class)
        %                preduction
        %  ____________________________
        %             |   pos     neg
        %   __________|_________________
        %   true | pos| 1,1 TP | 1,2 FN
        %   clas | neg| 2,1 FP | 2,2 TN
        
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
        
        mcc_denom = (TP+FP) * (TP+FN) * (TN+FP) * (TN + FN);
        
        % make sure denominator is not zero, wiki says set to 1
        if mcc_denom == 0
            mcc_denom = 1;
        end
        MCC = (TP * TN - FP * FN) ./ sqrt(mcc_denom);
        
        
        contab = [TP,FN,FP,TP];
        if any(contab < 5) || sum(contab) < 20
            MCC = 0;   %MCC is ill defined for small numbers ???
        end
        
    end
end

