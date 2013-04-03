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

