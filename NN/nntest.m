function [er, bad] = nntest(nn,nnff, x, y)
    labels = nnpredict(nn,nnff, x);
    [~, expected] = max(y,[],2);
    bad = find(labels ~= expected);    
    er = numel(bad) / size(x, 1);
end
