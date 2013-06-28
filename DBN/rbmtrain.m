function rbm = rbmtrain(rbm, x, opts)
    assert(isfloat(x), 'x must be a float');
    if (~isfield(opts,'cdn')), opts.cdn = 1; end;
    assert(opts.cdn >= 1, 'cdn must be integer >= 1!');
    
    m = size(x, 1);
    numbatches = floor(m / opts.batchsize);
    cdn = opts.cdn;
    
    for i = 1 : opts.numepochs
        kk = randperm(m);
        err = 0;
        for l = 1 : numbatches
            batch = extractminibatch(kk,l,opts.batchsize,x);
            batchsize = size(batch,1);  % actual batchsize (last batch may be larger then others)
            
            v = cell(cdn + 1,1);
            h = cell(cdn + 1,1);
            h_sample = cell(cdn + 1,1);
            
            % always (even last step, just don't use samples) sample hidden units
            % never sample visible units
            v{1} = batch;
            [h{1}, h_sample{1}] = rbmup(rbm,v{1});
                        
            for k = 2 : cdn + 1
                v{k} = rbmdown(rbm,h_sample{k-1});
                [h{k}, h_sample{k}] = rbmup(rbm,v{k});
                
            end;
            
            % use probabilities, not sampled values, for collecting statistics
            phase_pos = h{1}' * v{1};
            phase_neg = h{cdn + 1}' * v{cdn + 1};

            rbm.vW = rbm.momentum * rbm.vW + rbm.alpha * (phase_pos - phase_neg)     / batchsize;
            rbm.vb = rbm.momentum * rbm.vb + rbm.alpha * sum(v{1} - v{cdn + 1})' / batchsize;
            rbm.vc = rbm.momentum * rbm.vc + rbm.alpha * sum(h{1} - h{cdn + 1})' / batchsize;

            rbm.W = rbm.W + rbm.vW;
            rbm.b = rbm.b + rbm.vb;
            rbm.c = rbm.c + rbm.vc;

            err = err + sum(sum((v{1} - v{cdn + 1}) .^ 2)) / batchsize;
        end
        
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Average reconstruction error is: ' num2str(err / numbatches)]);
        
    end
end