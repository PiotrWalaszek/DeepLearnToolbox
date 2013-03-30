function nn = nnapplygrads_gpu(nn)
%NNAPPLYGRADS updates weights and biases with calculated gradients
% nn = nnapplygrads(nn) returns an neural network structure with updated
% weights and biases

% Evaluation of the normalization constant for the momentum term (different for each layer)
norm_constant=[];
if nn.momentum~=0
    if nn.normalize_momentum==1
        norm_dE=nan(1,nn.n-1);
        norm_vW=nan(1,nn.n-1);
        for ii = 1 : nn.n-1
            dE = nn.dW{ii} + nn.weightPenaltyL2 * nn.W{ii};
            norm_dE(ii) = norm(dE(:),2); % Vector (not matrix) norm of the error
            norm_vW(ii) = norm(nn.vW{ii}(:),2);
        end
        norm_constant=(nn.learningRate.*norm_dE)./norm_vW;
    else
        norm_constant=gpuArray.ones(size(nn.size(1:end-1)));
    end
end


for i = 1 : (nn.n - 1)
    if(nn.weightPenaltyL2>0)
        dW = nn.learningRate * (nn.dW{i} + nn.weightPenaltyL2 * nn.W{i});
    else
        dW = nn.learningRate * nn.dW{i};
    end
    db = nn.db{i};
    
    dW = nn.learningRate * dW;
    db = nn.learningRate * db;
    
    
    
    if(nn.momentum>0)
        nn.vW{i} = nn.momentum*nn.vW{i} + dW;
        
        if i ~= 1
            nn.vb{i} = nn.momentum*nn.vb{i} + db;
            db = nn.vb{i};
        end
        dW = nn.vW{i};
        
    end
    
    nn.W{i} = nn.W{i} - dW;
    if i ~= 1
        nn.b{i} = nn.b{i} - db;
    end
    
    %Max L2 norm of incoming weights to individual neurons
    if nn.weightMaxL2norm > 0;
        %Get the L2 norm indput to the individual Neurons
        L2_norm_input = sum(nn.W{i}.^2,2);
        for j = 1:nn.size(i+1) %loop through the neurons;
            if L2_norm_input(j) > nn.weightMaxL2norm
                nn.W{i}(j,:) = nn.W{i}(j,:)./sqrt(L2_norm_input(j)/nn.weightMaxL2norm);
            end
        end
    end
end
end