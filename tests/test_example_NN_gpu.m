
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
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

%% ex1 vanilla neural net
rng(0);
nn = nnsetup([784 1200 1200 1200 10]);
nn.output = 'softmax';
nn.activation_function = 'sigm';
nn.normalize_input = 0;

nn.dropoutFraction = 0.5;
nn.inputZeroMaskedFraction = 0.2;
%nn.weightPenaltyL2 = 1e-6;
nn.weightMaxL2norm = 15;
nn.cast                     = @double;
nn.caststr                  = 'double';
nn.errfun    = @nntest;

opts.plotfun = @nnplottest;
opts.numepochs              =  5000;   %  Number of full sweeps through data
opts.momentum_variable      = [linspace(0.5,0.95,1500 ) linspace(0.95,0.95,opts.numepochs -1500)];
opts.learningRate_variable  =  8.*(linspace(0.998,0.998,opts.numepochs ).^linspace(1,opts.numepochs,opts.numepochs ));
opts.learningRate_variable  = opts.learningRate_variable.*opts.momentum_variable;
opts.plot                   = 1;
opts.batchsize              = 1000;  %  Take a mean gradient step over this many samples
opts.ntrainforeval          = 5000; % number of training samples that are copied to the gpu and used to
% evalute training performance
% if you have a small dataset set this to number
% opts.plotfun = @nnplotmatthew;
% nn.errfun               = @nnmatthew;
% of samples in your training data
tt = tic;
[nn_gpu,L,loss] = nntrain_gpu(nn, train_x, train_y, opts);
toc(tt);
%[nn_cpu,L,loss] = nntrain(nn, train_x, train_y, opts);

[er_gpu, bad] = nntest(nn_gpu, test_x, test_y);
%[er_cpu, bad] = nntest(nn_cpu, test_x, test_y);
fprintf('Error: %f \n',er_gpu);
%fprintf('Error GPU (single); %f \n',er_cpu);

%
% load mnist_uint8;
% cast = @double;
%
% train_x = cast(train_x) / 255;
% test_x  = cast(test_x)  / 255;
% train_y = cast(train_y);
% test_y  = cast(test_y);
%
% % normalize
% [train_x, mu, sigma] = zscore(train_x);
% test_x = normalize(test_x, mu, sigma);
%
% %% ex1 vanilla neural net
% rng(0);
% nn = nnsetup([784 200 10]);
%
% fprintf('DOUBLE PRECISION PERFORMANCE \n')
% %[nn_gpu,L,loss] = nntrain_gpu(nn, train_x, train_y, opts);
% [nn_cpu,L,loss] = nntrain(nn, train_x, train_y, opts);
% %[er_gpu, bad] = nntest(nn_gpu, test_x, test_y);
% [er_cpu, bad] = nntest(nn_cpu, test_x, test_y);
% fprintf('Error GPU (double): %f \n',er_gpu);
% fprintf('Error GPU (double); %f \n',er_cpu);
