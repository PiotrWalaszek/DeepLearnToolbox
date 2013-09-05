function example_DBN

load eegP300_Subject8_2-4sesionForTrain_notBalance_Y0or1;

%% DBN parameters:
%  if default value is given, parameter may not be set in user code

opts.numepochs  = 70;           % number of epochs (full sweeps through data)
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

dbn.sizes       = [1500 2000];  % size of hidden layers

rand('state',0)

dbn = dbnsetup(dbn, train_x, opts);
dbn.rbm{1}.cdn = 3;
dbn = dbntrain(dbn, train_x, opts);

save('C:\Users\piow\Documents\DeepLearnToolbox\saves\dbnForSubject8_notBalance_s1500cd3-2000b.mat','dbn');

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 1);
nn.activation_function = 'sigm';
nn.output = 'linear';
%train nn
opts.numepochs = 30;
opts.batchsize = 10;
opts.learningRate_variable = linspace(0.3,0.2,opts.numepochs);
opts.momentum_variable = zeros(1,opts.numepochs) + 0.9;
%opts.dropoutFraction = 0.5;

nn = nntrain(nn, train_x, train_y', opts);

save('C:\Users\piow\Documents\DeepLearnToolbox\saves\nnForSubject8_notBalance_s1500cd3-2000b_1Output-linear.mat','nn');


predicted_score = nnpredict(nn, test_x)';

[X,Y,T,AUC,OPTROCPT] = perfcurve(test_y,predicted_score,1);
plot(X,Y);
hold on;
a = linspace(0.0,1.0,40);
plot(a,a,'g');
xlabel('1 - Specyficznosc'); 
ylabel('Czulosc')
title(strcat('Krzywa ROC dla modelu BLDA. AUC = ',num2str(AUC)));
hold off;

figure;
hist(predicted_score);

predicted_labels = predicted_score > 0.5;
einfo = error_info( predicted_labels, test_y);
fprintf('Matthews correlation coefficient = %f\n',einfo.mcc);
fprintf('Error = %f\n',einfo.error);
fprintf('Confusion matrix\nA \tPredicted class\n');
fprintf('c \t\tNo,\t\tYes\n');
fprintf('t No\t%d,\t%d\n',einfo.tn,einfo.fp);
fprintf('u Yes\t%d,\t%d\n',einfo.fn,einfo.tp);
fprintf('a\nl\n');


fprintf('Results for training data\n');
predicted_score = nnpredict(nn, train_x)';
predicted_labels = predicted_score > 0.5;

einfo = error_info(predicted_labels, train_y);
fprintf('Matthews correlation coefficient = %f\n',einfo.mcc);
fprintf('Error = %f\n',einfo.error);
fprintf('Confusion matrix\nA \tPredicted class\n');
fprintf('c \t\tNo,\t\tYes\n');
fprintf('t No\t%d,\t%d\n',einfo.tn,einfo.fp);
fprintf('u Yes\t%d,\t%d\n',einfo.fn,einfo.tp);
fprintf('a\nl\n');
