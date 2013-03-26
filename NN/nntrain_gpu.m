function [hnn, L,hloss]  = nntrain_gpu(hnn, htrain_x, htrain_y, opts, hval_x, hval_y)
%NNTRAIN trains a neural net on cpu
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.
%
% hVARNAME is a variable on the host
% dVARNAME is a varibale on the gpu device
if nargin < 7
    outputpath = 'best_nn_weights';
end

assert(nargin == 4 || nargin == 6,'number ofinput arguments must be 4 or 6')

hloss.train.e               = [];
hloss.train.e_errfun        = [];
hloss.val.e                 = [];
hloss.val.e_errfun          = [];

corrfoeff_old = -999999999;

if nargin == 6
    opts.validation = 1;
else
    opts.validation = 0;
end


fhandle = [];
if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
    %check if plotting function is supplied, else use nnupdatefigures
    if ~isfield(opts,'plotfun')
        opts.plotfun = @nnupdatefigures;
    end
    
end

if isfield(opts, 'outputfolder')
    save_nn_flag = 1;
else
    save_nn_flag = 0;
end

%variable momentum
if isfield(opts, 'momentum_variable')
    if length(opts.momentum_variable) ~= opts.numepochs
        error('opts.momentum_variable must specify a momentum value for each epoch ie length(opts.momentum_variable) == opts.numepochs')
    end
    var_momentum_flag = 1;
else
    var_momentum_flag = 0;
end

%variable learningrate
if isfield(opts, 'learningRate_variable')
    if length(opts.learningRate_variable) ~= opts.numepochs
        error('opts.learningRate_variable must specify a learninrate value for each epoch ie length(opts.learningRate_variable) == opts.numepochs')
    end
    var_learningRate_flag = 1;
else
    var_learningRate_flag = 0;
end






m = size(htrain_x, 1);

batchsize = opts.batchsize;
numepochs = opts.numepochs;

numbatches = floor(m / batchsize);

%assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');

L = zeros(numepochs*numbatches,1);
n = 1;


% COPY NETWORK TO DEVICE 
dnn = cpNNtoGPU(hnn);



for i = 1 : numepochs
    tic;
    %update momentum
    if var_momentum_flag
       nn.momentum = opts.momentum_variable(i);
    end
    %update learning rate
    if var_learningRate_flag
       nn.learningRate = opts.learningRate_variable(i);
    end    
    
    kk = randperm(m);
    for l = 1 : numbatches
        
        hbatch_x = extractminibatch(kk,l,batchsize,htrain_x);
        
        %Add noise to input (for use in denoising autoencoder)
        if(nn.inputZeroMaskedFraction ~= 0)
            hbatch_x = hbatch_x.*(rand(size(hbatch_x))>nn.inputZeroMaskedFraction);
        end
        
        
        % COPY BATCHES TO GPU DEVICE
        dbatch_x = gpuArray(hbatch_x);
        dbatch_y = gpuArray(extractminibatch(kk,l,batchsize,htrain_y));
        
        % use gpu functions to train
        dnn = nnff_gpu(dnn, dbatch_x, dbatch_y);
        dnn = nnbp_gpu(dnn);
        dnn = nnapplygrads_gpu(dnn);      
        L(n) = gather(dnn.L);        
        n = n + 1;
    end
    
    t = toc;
    
    
    % copy netowrk to host and evalute performance
    hnn = cpNNtoHost(dnn);
    
    %after each epoch update losses
    if opts.validation == 1
        hloss = nneval(hnn, hloss, htrain_x, htrain_y, hval_x, hval_y);
    else
        hloss = nneval(hnn, hloss, htrain_x, htrain_y);
    end
    

    
    % plot if figure is available
    if ishandle(fhandle)
        opts.plotfun(hnn, fhandle, hloss, opts, i);
    end
    
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds' '. Mean squared error on training set is ' num2str(mean(L((n-numbatches):(n-1))))]);
    
    %save model after every ten epochs if it is better than the previous
    %saved model
    if save_nn_flag && mod(i,10) == 0
       corrfoeff = nnmatthew(hnn, hval_x, hval_y);
       
       if corrfoeff(1) > corrfoeff_old
            epoch_nr = i;
            save([opts.outputfolder '.mat'],'hnn','opts','epoch_nr');
            disp(['Saved weights to: ' opts.outputfolder]);
            corrfoeff_old = corrfoeff(1);
       end
    end
    
end

clear dnn
clear dbatch_x
clear dbatch_y
end

