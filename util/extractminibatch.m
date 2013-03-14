function batch = extractminibatch(kk,l,batchsize,train_x)
%EXTRACTMINIBATCH extract minibatch
batch_start = (l - 1) * batchsize + 1;
batch_end   = l * batchsize;
n_samples = size(train_x,1);
if (batch_end + batchsize) <= n_samples
    
    %batch_x = train_x(  kk((l - 1) * batchsize + 1 : l * batchsize)  , :);
    batch = train_x(    kk(batch_start:batch_end),:);
else
    batch = train_x(    kk(batch_start:end),:);
end
end

