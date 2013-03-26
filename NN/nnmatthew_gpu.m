function [mcc, bad] = nnmatthew_gpu(nn, x, y)
%MATTHEW calculate matthew coefficient for all target classes
%   Calculates the matthew correlation coefficient for all target calasses.
%   for a n-class classification problem the function returns a n
%   dimensional row vector.
%   bad is notused, but returned for compability with rest of code. 
bad = [];
n_output = size(y,2);

assert(n_output~=1,'Behavior of matthew correlation not tested with single output')

% predict labels with network 
pred = nnpredict_gpu(nn, x); 

% find correct targets
[~, expected] = max(y,[],2);  

% testing
%expected = [1,2,1,1,2,3,4];
%labels   = [1,2,1,1,2,3,2];
mcc = gpuArray(zeros(1,n_output));
for target_class = 1:n_output    % testing: set to four    
    
    %create binary vectors for each class. For each class (target_class)
    % match the predition with target class and the expected class with the
    % target class
    pred_class = ~(pred     == target_class);
    true_class = ~(expected == target_class);
    
    %
    [~,~,~,~,MCC] =  matthew_calc(pred_class,true_class);
    mcc(target_class) = MCC;
    
    %   testing
    %     disp([target_class,MCC])
    %     disp([pred_class;
    %     true_class])
end
   
   
       function [TP,TN,FP,FN,MCC] =  matthew_calc(pred_class,true_class)
        TP = sum( (pred_class == true_class) .* (true_class == 0) ); %True positive
        TN = sum( (pred_class == true_class) .* (true_class == 1) ); %True negative
        FP = sum( (pred_class ~= true_class) .* (pred_class == 1) ); %False positive
        FN = sum( (pred_class ~= true_class) .* (pred_class == 0) ); %False negative  
        
        
%         [~,x2] = chisquarecont(contab);
%         MCC = abs(sqrt(x2./ n_samples));
        mcc_denom = (TP+FP) * (TP+FN) * (TN+FP) * (TN + FN);
       
        % make sure denominator is not zero, wiki says set to 1 
        if mcc_denom == 0
            mcc_denom = 1;
        end
        MCC = (TP * TN - FP * FN) ./ sqrt(mcc_denom);
        
        
        contab = [TP,FP,FN,TP];
        if any(contab < 5) || sum(contab) < 20
            MCC = 0;   %MCC is ill defined for small numbers ???
        end
        
    end
end

