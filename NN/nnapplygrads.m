function nn = nnapplygrads(nn)
%NNAPPLYGRADS updates weights and biases with calculated gradients
% nn = nnapplygrads(nn) returns an neural network structure with updated
% weights and biases
    
    for i = 1 : (nn.n - 1)
        if(nn.weightPenaltyL2>0)
            dW = nn.dW{i} + nn.weightPenaltyL2 * nn.W{i};
        else
            dW = nn.dW{i};
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
            L2 = nn.weightMaxL2norm;
            L2_norm_input = sum(nn.W{i}.^2,2);
            norm_factor = sqrt(L2_norm_input/L2);
            idx = norm_factor < 1;
            norm_factor(idx) = 1;
            nn.W{i} = bsxfun(@rdivide,nn.W{i},norm_factor);    
        end
        
        
    end
end
