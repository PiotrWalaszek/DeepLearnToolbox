function [ dnn ] = cpNNtoGPU( hnn,cast)
fld = fields(hnn);
for i=1:numel(fld)
    fieldName = fld{i};
    switch fieldName
        case 'W'
            for j=1:numel(hnn.W)
                dnn.W{j} = gpuArray(cast(hnn.W{j}));
            end
        case 'vW'
            for j=1:numel(hnn.vW)
                dnn.vW{j} = gpuArray(cast(hnn.vW{j}));
            end
        case 'p'
            for j=1:numel(hnn.p)
                dnn.p{j} = gpuArray(cast(hnn.p{j}));
            end
        case 'dW'
            for j=1:numel(hnn.dW)
                dnn.dW{j} = gpuArray(cast(hnn.dW{j}));
            end
        case 'e'
            for j=1:numel(hnn.e)
                dnn.e{j} = gpuArray(cast(hnn.e{j}));
            end
        case 'a'
            for j=1:numel(hnn.a)
                dnn.a{j} = gpuArray(cast(hnn.a{j}));
            end
        otherwise
            dnn.(fieldName) = hnn.(fieldName);
    end
end