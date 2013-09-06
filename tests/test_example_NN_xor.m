function test_example_NN_xor
x = [0 0; 1 0; 0 1; 1 1];
y = [0; 1; 1; 0]; 
nn = nnsetup([2 2 1]); % 2 input, 2 hidden and 1 output unit
nn.activation_function = 'sigm';
%nn.b{1} = [-1; -1]; % bias for hidden units
%nn.b{2} = -0.1; % bias for output unit
%nn.W{1} = [1 -1; -1 1]; %weights between input and hidden layer 
%nn.W{2} = [1 1]; % weights between hidden and output layer
nn.isGPU = 0; % compute on cpu
opts.numepochs = 300;
opts.batchsize = 1;
opts.learningRate_variable = ones(opts.numepochs); %learning rate 1 to be simple
opts.momentum_variable = zeros(opts.numepochs); %momentum 0

nn = nntrain(nn, x, y, opts);
predicted_score = nnpredict(nn, x);
predict = predicted_score > 0.5; % threshold
wrong = y ~= predict;
if sum(wrong) == 0
	fprintf('Pass!');
else
	fprintf('Faill!');
end
end