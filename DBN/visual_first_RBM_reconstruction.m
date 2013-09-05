function visual_first_gRBM_reconstruction
load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);


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

dbn.sizes = [1000];

dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);
%save('C:\Users\piow\Documents\DeepLearnToolbox\saves\rbm_mnist_s1000_cd1_e70.mat','dbn');

figure;
weights = visualize(dbn.rbm{1}.W');
weights_image = imagesc(weights, [min(min(weights)) max(max(weights))]);
axis equal;
colormap gray;
%saveas(weights_image,'C:\Users\piow\Documents\DeepLearnToolbox\saves\reconstruction_linear\wagi','png');

for i = 1:150
    v0 = test_x(i,:);
    y = find(test_y(i,:),1);
    
    im = vector2image(v0);
    subplot(1,2,1);
    imagesc(im, [min(min(im)) max(max(im))]);
    axis equal;
    colormap gray;

    [h0 hs0] = rbmup(dbn.rbm{1},v0);
    [v0 vs0] = rbmdown(dbn.rbm{1},h0);
    
    im = vector2image(v0);
    subplot(1,2,2);
    reconstruction = imagesc(im, [min(min(im)) max(max(im))]);
    axis equal;
    colormap gray;
    
    %saveas(reconstruction,strcat('C:\Users\piow\Documents\DeepLearnToolbox\saves\reconstruction_linear\',int2str(i),'_',int2str(y)),'png');
end