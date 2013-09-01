function example_DBN

load eegP300YVectorBalanceTrunc;

%% DBN parameters:
%  if default value is given, parameter may not be set in user code

opts.numepochs  = 1;           % number of epochs (full sweeps through data)
opts.batchsize  = 10;           % number of traning examples to average gradient over (one mini-batch size)
                                % (set to size(train_x,1) to perform full-batch learning)
opts.momentum   = 0.5;          % learning momentum (default: 0)
opts.momentum_final = 0.9;  % learning momentum for first RBM (default: learning momentum)
opts.alpha      = 0.1;          % learning rate
opts.alpha_first_rbm = 0.1;   % learning rate for first RBM (default: learning rate)
opts.cdn        = 1;            % number of steps for contrastive divergence learning (default: 1)
opts.vis_units  = 'sigm';     % type of visible units (default: 'sigm')
opts.hid_units  = 'sigm';       % type of hidden units  (default: 'sigm')
                                % units can be 'sigm' - sigmoid, 'linear' - linear
                                % 'NReLU' - noisy rectified linear (Gaussian noise)

dbn.sizes       = [1500 2000 3000];  % size of hidden layers

rand('state',0)

dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

%save('C:\Users\piow\Documents\DeepLearnToolbox\saves\dbnTrunc.mat','dbn');

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 2);
nn.activation_function = 'sigm';

%train nn
opts.numepochs = 1;
opts.batchsize = 10;
opts.learningRate_variable = linspace(1.0,0.3,opts.numepochs);
opts.momentum_variable = zeros(1,opts.numepochs) + 0.9;
opts.dropoutFraction = 0.5;
opts.output = 'softmax';

nn = nntrain(nn, train_x, train_y, opts);

save('C:\Users\piow\Documents\DeepLearnToolbox\saves\nnTrunc.mat','nn');

error_info = nntest_extended(nn, test_x, test_y);
fprintf('Matthews correlation coefficient = %f\n',error_info.mcc);
fprintf('Error = %f\n',error_info.error);
fprintf('Confusion matrix\nA \tPredicted class\n');
fprintf('c \t\tNo,\t\tYes\n');
fprintf('t No\t%d,\t%d\n',error_info.tn,error_info.fp);
fprintf('u Yes\t%d,\t%d\n',error_info.fn,error_info.tp);
fprintf('a\nl\n');

fprintf('Results for training data\n');
error_info = nntest_extended(nn, train_x, train_y);
fprintf('Matthews correlation coefficient = %f\n',error_info.mcc);
fprintf('Error = %f\n',error_info.error);
fprintf('Confusion matrix\nA \tPredicted class\n');
fprintf('c \t\tNo,\t\tYes\n');
fprintf('t No\t%d,\t%d\n',error_info.tn,error_info.fp);
fprintf('u Yes\t%d,\t%d\n',error_info.fn,error_info.tp);
fprintf('a\nl\n');
