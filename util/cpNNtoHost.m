function [ hnn ] = cpNNtoHost( dnn )
fld = fields(dnn);
for i=1:numel(fld)
    fieldName = fld{i};
    switch fieldName
        case 'W'
            for j=1:numel(dnn.W)
                hnn.W{j} = gather(dnn.W{j});
            end
        case 'vW'
            for j=1:numel(dnn.vW)
                hnn.vW{j} = gather(dnn.vW{j});
            end
        case 'p'
            for j=1:numel(dnn.p)
                hnn.p{j} = gather(dnn.p{j});
            end
        otherwise
            hnn.(fieldName) = dnn.(fieldName);
    end
end