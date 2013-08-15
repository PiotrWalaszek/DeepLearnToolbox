function example_DBN

load eegP300;

train_x = double(train_x);% / 255;
test_x  = double(test_x);%  / 255;
train_y = double(train_y);
test_y  = double(test_y);

%% DBN parameters:
%  if default value is given, parameter may not be set in user code

opts.numepochs  = 1;           % number of epochs (full sweeps through data)
opts.batchsize  = 10;           % number of traning examples to average gradient over (one mini-batch size)
                                % (set to size(train_x,1) to perform full-batch learning)
opts.momentum   = 0.5;          % learning momentum (default: 0)
opts.momentum_final = 0.9;  % learning momentum for first RBM (default: learning momentum)
opts.alpha      = 0.1;          % learning rate
opts.alpha_first_rbm = 0.001;   % learning rate for first RBM (default: learning rate)
opts.cdn        = 1;            % number of steps for contrastive divergence learning (default: 1)
opts.vis_units  = 'linear';     % type of visible units (default: 'sigm')
opts.hid_units  = 'sigm';       % type of hidden units  (default: 'sigm')
                                % units can be 'sigm' - sigmoid, 'linear' - linear
                                % 'NReLU' - noisy rectified linear (Gaussian noise)

dbn.sizes       = [1000 1000 2000];  % size of hidden layers

rng(0);

dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 2);
nn.activation_function = 'sigm';

%train nn
opts.numepochs =  1;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);

%assert(er < 0.10, 'Too big error');
fprintf('error = %f\n',er);
disp(bad)

