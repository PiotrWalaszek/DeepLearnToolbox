
DeepLearnToolbox
================

A Matlab toolbox for Deep Learning.

Deep Learning is a new subfield of machine learning that focuses on learning deep hierarchical models of data.
It is inspired by the human brain's apparent deep (layered, hierarchical) architecture.
A good overview of the theory of Deep Learning theory is
[Learning Deep Architectures for AI](http://www.iro.umontreal.ca/~bengioy/papers/ftml_book.pdf)

For a more informal introduction, see the following videos by Geoffrey Hinton and Andrew Ng.

* [The Next Generation of Neural Networks](http://www.youtube.com/watch?v=AyzOUbkUf3M) (Hinton, 2007)
* [Recent Developments in Deep Learning](http://www.youtube.com/watch?v=VdIURAu1-aU) (Hinton, 2010)
* [Unsupervised Feature Learning and Deep Learning](http://www.youtube.com/watch?v=ZmNOAtZIgIk) (Ng, 2011)

If you use this toolbox in your research please cite [Prediction as a candidate for learning deep hierarchical models of data](http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6284)

Directories included in the toolbox
-----------------------------------

`NN/`   - A library for Feedforward Backpropagation Neural Networks (with GPU version)

`CNN/`  - A library for Convolutional Neural Networks (CPU only)

`DBN/`  - A library for Deep Belief Networks (CPU only)

`SAE/`  - A library for Stacked Auto-Encoders (CPU only)

`CAE/` - A library for Convolutional Auto-Encoders (CPU only)

`util/` - Utility functions used by the libraries

`data/` - Data used by the examples

`tests/` - unit tests to verify toolbox is working

For references on each library check REFS.md

Setup
-----

1. Download.
2. addpath(genpath('DeepLearnToolbox'));

Everything is work in progress
------------------------------

Example: Deep Belief Network
---------------------
```matlab

function test_example_DBN
load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

%% DBN parameters:
%  if default value is given, parameter may not be set in user code

opts.numepochs  = 10;       % number of epochs (full sweeps through data)
opts.batchsize  = 100;      % number of traning examples to average gradient over (one mini-batch size)
                            % (set to size(train_x,1) to perform full-batch learning)
opts.momentum   = 0;        % learning momentum (default: 0)
opts.alpha      = 1;        % learning rate
opts.cdn        = 1;        % number of steps for contrastive divergence learning (default: 1)
opts.vis_units  = 'sigm';   % type of visible units (default: 'sigm')
opts.hid_units  = 'sigm';   % type of hidden units  (default: 'sigm')
                            % units can be 'sigm' - sigmoid, 'linear' - linear
                            % 'NReLU' - noisy rectified linear (Gaussian noise)

dbn.sizes       = [10 20];  % size of hidden layers

%%  ex1 train a 100 hidden unit RBM and visualize its weights
rng('default'),rng(0);
dbn.sizes = [100];
opts.numepochs =   1;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;    
opts.cdn       =   1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);
figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

% Use code like this to visualize non-square images:
% X = dbn.rbm{1}.W';
% vert_size = 28;
% hor_size = 28;
% figure; visualize(X, [min(X(:)) max(X(:))], vert_size, hor_size);

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
rng(0);
%train dbn
dbn.sizes = [100 100];
opts.numepochs =   1;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 10);
nn.activation_function = 'sigm';

%train nn
opts.numepochs =  1;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);

assert(er < 0.10, 'Too big error');

```


Example: Stacked Auto-Encoders
---------------------
```matlab

function test_example_SAE
load mnist_uint8;

train_x = double(train_x)/255;
test_x  = double(test_x)/255;
train_y = double(train_y);
test_y  = double(test_y);

%%  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
rng(0);
sae = saesetup([784 100]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 1;
sae.ae{1}.inputZeroMaskedFraction   = 0.5;
opts.numepochs =   1;
opts.batchsize = 100;
sae = saetrain(sae, train_x, opts);
visualize(sae.ae{1}.W{1}(:,2:end)')

% Use the SDAE to initialize a FFNN
nn = nnsetup([784 100 10]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 1;
nn.W{1} = sae.ae{1}.W{1};

% Train the FFNN
opts.numepochs =   1;
opts.batchsize = 100;
[nn,L,loss] = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);
assert(er < 0.16, 'Too big error');

%% ex2 train a 100-100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
rng(0);
sae = saesetup([784 100 100 100]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 1;
sae.ae{1}.inputZeroMaskedFraction   = 0.5;

sae.ae{2}.activation_function       = 'sigm';
sae.ae{2}.learningRate              = 1;
sae.ae{2}.inputZeroMaskedFraction   = 0.5;

opts.numepochs =   1;
opts.batchsize = 100;
sae = saetrain(sae, train_x, opts);
visualize(sae.ae{1}.W{1}(:,2:end)')

% Use the SDAE to initialize a FFNN
nn = nnsetup([784 100 100 10]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 1;

%add pretrained weights
nn.W{1} = sae.ae{1}.W{1};
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   1;
opts.batchsize = 100;
[nn,L,loss] = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);
assert(er < 0.1, 'Too big error');

```


Example: Convolutional Neural Nets
---------------------
```matlab

function test_example_CNN
load mnist_uint8;

train_x = double(reshape(train_x',28,28,60000))/255;
test_x = double(reshape(test_x',28,28,10000))/255;
train_y = double(train_y');
test_y = double(test_y');

%% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error
rng(0)
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};
cnn = cnnsetup(cnn, train_x, train_y);

opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs = 1;

cnn = cnntrain(cnn, train_x, train_y, opts);

[er, bad] = cnntest(cnn, test_x, test_y);

%plot mean squared error
figure; plot(cnn.rL);

assert(er<0.12, 'Too big error');

```


Example: Neural Networks
---------------------
```matlab

clear all
load mnist_uint8;



val_x = double(train_x) / 255;
train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;

val_y = double(train_y);
train_y = double(train_y);
test_y  = double(test_y);

% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

%% NN creation parameters:
%  default values given, set parameter manually only if you want to change it
%  set paramters AFTER calling nnsetup()

nn.activation_function              = 'tanh_opt';   %  Activation functions of hidden layers: 
                                                    % 'sigm' (sigmoid), 'tanh_opt' (optimal tanh), 'ReLU' (rectified linear)
nn.learningRate                     = 2;            %  learning rate 
                                                    % Note: typically needs to be lower when using 'sigm'
                                                    % activation function and non-normalized inputs.
nn.momentum                         = 0.5;          %  Momentum
nn.weightPenaltyL2                  = 0;            %  L2 regularization
nn.weightMaxL2norm                  = 0;            %  Max L2 norm of incoming weights to individual Neurons - see Hinton 2009 dropout paper            
nn.nonSparsityPenalty               = 0;            %  Non sparsity penalty
nn.sparsityTarget                   = 0.05;         %  Sparsity target (used only if SpaisityPenalty ~= 0)
nn.inputZeroMaskedFraction          = 0;            %  Used for Denoising AutoEncoders
nn.dropoutFraction                  = 0;            %  Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
nn.output                           = 'sigm';       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'
nn.errfun                           = [];           %  Empty for standard error options: @nnmatthew, @nnmatthew_gpu
                                                    % @nntest is used for @nnplotnntest
                                                    % @nnsigp is used for SignalP networks

%% NN training paramters (opts):
%  default values, may be set by line:
%  opts = nnopts_setup; 

opts.validation             = 1;    % Is overruled by the number of arguments in nntrain - 
                                    % ie. nntrain must have 6 input arguments for opts.validation = 1
opts.plot                   = 0;    % Plots the training progress if set
opts.plotfun                = @nnupdatefigures; 
                                    % Plots network error, alternatives:
                                    % @nnplotmatthew (plots error and matthew coefficients for each class)
                                    % @nnplotnntest (plots error and misclassification rate)
                                    % @nnplotsigp (used with SignalP networks)
opts.outputfolder           = '';   % If set the network is saved to the path specified by outpufolder after every 100 epochs. 
                                    % If plot is enabled the figure is also saved here after every 10 epochs.
opts.learningRate_variable  = [];   % If set specifies a momentum for every epoch. 
                                    % ie length(opts.momentum) == opts.numepochs.
opts.momentum_variable      = [];   % If set specifies a learning rate for every epoch.
                                    % ie length(opts.learningRate_variable) == opts.numepochs.
opts.numepochs              = 1;    % Number of epochs (runs through the complete training data set)
opts.batchsize              = 100;  % Number of traning examples to average gradient over (one mini-batch size)
                                    % (set to size(train_x,1) to perform full-batch learning)
opts.ntrainforeval          = [];   % Only relevant for GPU training. Sets the number of evaluation training datasets to use. 
                                    % Set this parameter to something small if you run into memory problems

%% ex1 vanilla neural net
rng(0);
nn = nnsetup([784 100 10]);
nn.weightMaxL2norm                  = 15;            %  Max L2 norm of incoming weights to individual Neurons - see Hinton 2009 dropout paper            
opts = nnopts_setup;      % Default training options
opts.learningRate_variable = [linspace(10,0.01,20)];
opts.momentum_variable = [linspace(0.5,0.99,20)];
opts.plot = 1;
opts.numepochs =  20;   %  Number of full sweeps through data
opts.batchsize = 100;  %  Take a mean gradient step over this many samples

[nn,L,loss] = nntrain(nn, train_x, train_y, opts, val_x,val_y);

[er, bad] = nntest(nn, test_x, test_y);

assert(er < 0.1, 'Too big error');

% Make an artificial one and verify that we can predict it
x = zeros(1,28,28);
x(:, 14:15, 6:22) = 1;
x = reshape(x,1,28^2);
figure; visualize(x');
predicted = nnpredict(nn,x)-1;

assert(predicted == 1);
%% ex2 neural net with L2 weight decay
rng(0);
nn = nnsetup([784 100 10]);

nn.weightPenaltyL2 = 1e-4;  %  L2 weight decay
opts = nnopts_setup;
opts.numepochs =  1;        %  Number of full sweeps through data
opts.batchsize = 100;       %  Take a mean gradient step over this many samples


[nn,L,loss] = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);
assert(er < 0.1, 'Too big error');


%% ex3 neural net with dropout
rng(0);
nn = nnsetup([784 100 10]);

nn.dropoutFraction = 0.5;   %  Dropout fraction 
opts.numepochs =  1;        %  Number of full sweeps through data
opts.batchsize = 100;       %  Take a mean gradient step over this many samples

[nn,L,loss] = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);
assert(er < 0.1, 'Too big error');

%% ex4 neural net with sigmoid activation function
rng(0);
nn = nnsetup([784 100 10]);

nn.activation_function = 'sigm';    %  Sigmoid activation function
nn.learningRate = 1;                %  Sigm require a lower learning rate
opts.numepochs =  1;                %  Number of full sweeps through data
opts.batchsize = 100;               %  Take a mean gradient step over this many samples

[nn,L,loss] = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);
assert(er < 0.1, 'Too big error');

%% ex5 plotting functionality
rng(0);
nn = nnsetup([784 20 10]);
opts.numepochs         = 5;            %  Number of full sweeps through data
nn.output              = 'softmax';    %  use softmax output
opts.batchsize         = 1000;         %  Take a mean gradient step over this many samples
opts.plot              = 1;            %  enable plotting

[nn,L,loss] = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);
assert(er < 0.1, 'Too big error');

%% ex6 neural net with sigmoid activation and plotting of validation and training error
% split training data into training and validation data
vx   = train_x(1:10000,:);
tx = train_x(10001:end,:);
vy   = train_y(1:10000,:);
ty = train_y(10001:end,:);

%%
rng(0);
nn.activation_function  = 'sigm';
nn                      = nnsetup([784 200 10]);     
nn.output               = 'softmax';                   %  use softmax output
nn.errfun               = @nnmatthew;     
nn.weightPenaltyL2      = 1e-4;
opts.numepochs          = 30;  
opts.learningRate_variable = ones(1,opts.numepochs);
opts.momentum_variable     = 0.5*ones(1,opts.numepochs);
opts.numepochs          = 30;                           %  Number of full sweeps through data
opts.batchsize          = 1000;                        %  Take a mean gradient step over this many samples

opts.plot               = 1;                           %  enable plotting

%the default for errfun is nntest, the default for plotfun is updatefigures
                 %  This function is applied to train and optionally validation set should be format [er, notUsed] = name(nn, x, y)
opts.plotfun                = @nnplotmatthew;

[nn,L,loss] = nntrain(nn, tx, ty, opts, vx, vy);                %  nntrain takes validation set as last two arguments (optionally)


[er, bad] = nntest(nn, test_x, test_y);
assert(er < 0.1, 'Too big error');
```


