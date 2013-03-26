addpath('../data');
addpath('../util');
addpath('../NN');
load mnist_uint8;

cast = @single;

train_x = cast(train_x) / 255;
test_x  = cast(test_x)  / 255;
train_y = cast(train_y);
test_y  = cast(test_y);

% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

%% ex1 vanilla neural net
rng(0);
nn = nnsetup([784 200 10]);

fprintf('SINGLE PRECISION PERFORMANCE \n')
opts.numepochs =  10;   %  Number of full sweeps through data
opts.batchsize = 1000;  %  Take a mean gradient step over this many samples
[nn_gpu,L,loss] = nntrain_gpu(nn, train_x, train_y, opts);
[nn_cpu,L,loss] = nntrain(nn, train_x, train_y, opts);
[er_gpu, bad] = nntest(nn_gpu, test_x, test_y);
[er_cpu, bad] = nntest(nn_cpu, test_x, test_y);
fprintf('Error GPU (single): %f \n',er_gpu);
fprintf('Error GPU (single); %f \n',er_cpu);


load mnist_uint8;
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
nn = nnsetup([784 200 10]);

fprintf('DOUBLE PRECISION PERFORMANCE \n')
[nn_gpu,L,loss] = nntrain_gpu(nn, train_x, train_y, opts);
[nn_cpu,L,loss] = nntrain(nn, train_x, train_y, opts);
[er_gpu, bad] = nntest(nn_gpu, test_x, test_y);
[er_cpu, bad] = nntest(nn_cpu, test_x, test_y);
fprintf('Error GPU (double): %f \n',er_gpu);
fprintf('Error GPU (double); %f \n',er_cpu);
