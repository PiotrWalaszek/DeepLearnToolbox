function [ dnn ] = cpNNtoGPU( hnn )
fld = fields(hnn);
for i=1:numel(fld)
    fieldName = fld{i};
    switch fieldName
        case 'W'
            for j=1:numel(hnn.W)
                dnn.W{j} = gpuArray(hnn.W{j});
            end
        case 'vW'
            for j=1:numel(hnn.vW)
                dnn.vW{j} = gpuArray(hnn.vW{j});
            end
        case 'p'
            for j=1:numel(hnn.p)
                dnn.p{j} = gpuArray(hnn.p{j});
            end
        otherwise
            dnn.(fieldName) = hnn.(fieldName);
    end
end

