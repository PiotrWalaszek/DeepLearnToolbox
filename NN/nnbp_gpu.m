function nn = nnbp_gpu(nn)
%NNBP performs backpropagation
% nn = nnbp(nn) returns an neural network structure with updated weights 
    
    n = nn.n;
    sparsityError = 0;
    switch nn.output
        case 'sigm'
            d_act = - nn.e .* (nn.a{n} .* (1 - nn.a{n}));
            d{n} = d_act;
            bd{n}   = single(ones(1,size(nn.e,1))) * d_act; 
        case {'softmax','linear'}
            d{n} = - nn.e;
            bd{n}   = single(ones(1,size(nn.e,1))) * -nn.e;
    end
    for i = (n - 1) : -1 : 2
        % Derivative of the activation function
        switch nn.activation_function 
            case 'sigm'
                d_act = nn.a{i} .* (gpuArray.ones(size(nn.a{i})) - nn.a{i});
            case 'ReLU'  % linear rectified units max(0,x) 
                d_act =nn.a{i} .* (nn.a{i}>0);
            case 'tanh_opt'
                d_act = 1.7159 * 2/3 * (gpuArray.ones(size(nn.a{i})) - 1/(1.7159)^2 * nn.a{i}.^2);
            
        end
        
        if(nn.nonSparsityPenalty>0)
            pi = repmat(nn.p{i}, size(nn.a{i}, 1), 1);
            sparsityError = [gpuArray.zeros(size(nn.a{i},1),1) nn.nonSparsityPenalty * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi))];
        end
        
        % Backpropagate first derivatives
        if i+1==n % in this case in d{n} there is not the bias term to be removed             
            d{i} = (d{i + 1} * nn.W{i} + sparsityError) .* d_act; % Bishop (5.56)
            bd{i} = (bd{i + 1} * nn.b{i} + sparsityError) .* d_act;
        else % in this case in d{i} the bias term has to be removed
            d{i} = (d{i + 1}(:,2:end) * nn.W{i} + sparsityError) .* d_act;
        end
        clear d_act
        
        if(nn.dropoutFraction>0)
           d{i} = d{i} .* nn.dropOutMask{i};
        end

    end

    for i = 1 : (n - 1)
        if i+1==n
            nn.dW{i} = (d{i + 1}' * nn.a{i}) / size(d{i + 1}, 1);
            nn.db{i} = (bd{i + 1}'; 
        else
            nn.dW{i} = (d{i + 1}' * nn.a{i}) / size(d{i + 1}, 1);      
        end
    end
    clear d
    for  u = 1:n
        nn.a{u} = [];
    end
end