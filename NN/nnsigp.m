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
    
    [TP,TN,FP,FN ] =  calcconfusion(pred_class,true_class);
    confusionmat(:,:,target_class) = [TP FP; FN TN];
    
    
end

err(1) = matthew(confusionmat(:,:,1));       % 1) signalpeptide(1) MCC
err(2) = specificity(confusionmat(:,:,2));   % 2) Cleavage site(2) specificity
err(3) = specificity(confusionmat(:,:,2));   % 3) Cleavage site(2) precision
err(4) = specificity(confusionmat(:,:,2));   % 4) Cleavage site(2) MCC
err(5) = precision(confusionmat(:,:,3));     % 5) transmembrane(3) MCC




    function [TP,TN,FP,FN ] =  calcconfusion(pred_class,true_class)
        TP = sum( (pred_class == true_class) .* (true_class == 0) ); %True positive
        TN = sum( (pred_class == true_class) .* (true_class == 1) ); %True negative
        FP = sum( (pred_class ~= true_class) .* (pred_class == 1) ); %False positive
        FN = sum( (pred_class ~= true_class) .* (pred_class == 0) ); %False negative
        
    end

    function prec = precision(confusion)
        %calculate precision
        tp   = confusion(1,1);
        fp   = confusion(1,2);
        %fn   = conconfusion(2,1);
        %tn   = conconfusion(2,2);
        prec = tp / (tp + fp);
    end

    function spec = specificity(confusion)
        % calculates specifity
        fp   = confusion(1,2);
        %fn   = conconfusion(2,1);
        tn   = confusion(2,2);
        spec = tn / (fp + tn);
        
        
    end
    function mcc =  matthew(confusion)
        % claculates matthew correlation
        tp = confusion(1,1);
        fp = confusion(1,2);
        fn = confusion(2,1);
        tn = confusion(2,2);
        
        
        %check if mcc denominator is belew zero, set to 1 if so
        mcc_denom = (tp+fp) * (tp+fn) * (tn+fp) * (tn + fn);
        if mcc_denom == 0
            mcc_denom = 1;
        end
        
        mcc = (tp * tn - fp * fn) ./ sqrt(mcc_denom);
        
        % set mcc to zero if any entries in the conf mat is below 5
        if any(confusion(:) < 5) || sum(confusion) < 20
            mcc = 0;   %MCC is ill defined for small numbers ???
        end
    end

end


