function test_example_NN_gpu()
%% TEST_EXAMPLE_NN_GPU
% demonstrates performance on mnist dataset
% uses
%   -   L2 norm weight constraint, see hinton 2012 (http://arxiv.org/pdf/1207.0580.pdf)
%   -   Dropout 50 %
%   -   Dropout of input 20 %
%   -   Increasing  Momentum 
%   -   Decreasing momentum
%
% Also demonstrates the plotting functionality:
% You can create your own error functions and plot then.
% Error function must be of the format: [er, bad] = funname(nn, x, y)
%           - bad is a dummy included, set it to []
%           - err is  1 X D ROW vector of the calculated errors
%
% The plotting functino has the format:
%     funname(nn,fhandle,L,opts,i)
%       i       : number of current epoch
%       fhandle : handle to the plotting figure
%       L       : loss struct where
%                   * L.train.e [i x D] vector of taining set errors
%                   * L.val.e [i x D] vector of val set errors (optional, if val data is supplied)
%                   * L.train.e_errfun [i x D] vector of taining set errors from
%                       supplied error function
%                   * L.val.e_errfun [i x D] vector of val set errors from
%                       supplied error function  (Optional if val data is supplied)
%
% Note that the networks produced these files are not compatible with the
% original deeplearning toolbox including the RBM's because of w,b
% notation. To use RBM's use the original nnsetup function and afterwards: 
%  add b to w like nnRBM.w{1} = [nn.b{1} nn.W{w}] for all layers
addpath('../data');
addpath('../util');
addpath('../NN');
load mnist_uint8;
close all
cast = @double;

train_x = cast(train_x) / 255;
test_x  = cast(test_x)  / 255;
train_y = cast(train_y);
test_y  = cast(test_y);

% normalize
[train_x, mu, sigma]    = zscore(train_x);
test_x                  = normalize(test_x, mu, sigma);

rng(0);
nn                          = nnsetup([784 1200 1200 1200 10]);
nn.output                   = 'softmax'; % output function: softmax | sigm | linear
nn.activation_function      = 'sigm';    % activation func: sigm | tanh_opt | linear
nn.dropoutFraction          = 0.5;       % Droupout of hidden layers
nn.inputZeroMaskedFraction  = 0.2;       % input dropout
%nn.weightPenaltyL2         = 1e-6;      % weightdecay
nn.weightMaxL2norm          = 15;        % L" norm of incoming weights to each neuron are constrined to be below this value, rescaled if above
nn.cast                     = @double;   % double or single precision, single cuts memory usage by app. 50%
nn.caststr                  = 'double';  % double or single precision, single cuts memory usage by app. 50%
nn.errfun                   = @nntest;

opts.plotfun                = @nnplottest;
opts.numepochs              =  5000;        %  Number of full sweeps through data
opts.momentum_variable      = [linspace(0.5,0.95,1500 ) linspace(0.95,0.95,opts.numepochs -1500)];
opts.learningRate_variable  =  8.*(linspace(0.998,0.998,opts.numepochs ).^linspace(1,opts.numepochs,opts.numepochs ));
opts.learningRate_variable  = opts.learningRate_variable.*opts.momentum_variable;
opts.plot                   = 1;            % 0 = no plotting, migth speed up calc if epochs run fast
opts.batchsize              = 1000;         % Take a mean gradient step over this many samples. GPU note: below 500 is slow on GPU because of memory transfer
opts.ntrainforeval          = 5000;         % number of training samples that are copied to the gpu and used to evalute training performance
opts.outputfolder           = 'nns/hinton'; % saves network each 100 epochs and figures after 10. hinton is prefix to the files. 
                                            % nns is the name of a folder
                                            % from where this script is
                                            % called (probably tests/nns)
                                        

tt = tic;
[nn,L,loss]                 = nntrain_gpu(nn, train_x, train_y, opts,test_x,test_y); %use nntrain to train on cpu
toc(tt);
[er_gpu, bad]               = nntest(nn, test_x, test_y);    
fprintf('Error: %f \n',er_gpu);
