function dbn = dbnsetup(dbn, x, opts)

    rand_weight_sigma = 0.01;

    n = size(x, 2);
    dbn.sizes = [n, dbn.sizes];
    
    %defaults
    if (~isfield(opts,'vis_units') || isempty(opts.vis_units))
        opts.vis_units = 'sigm';
    end;
    if (~isfield(opts,'hid_units') || isempty(opts.hid_units))
        opts.hid_units = 'sigm';
    end;

    for u = 1 : numel(dbn.sizes) - 1
        dbn.rbm{u}.alpha    = opts.alpha;
        dbn.rbm{u}.momentum = opts.momentum;
        
        % make vis_units only actually visible units (1st layer)
        if (u == 1)
            dbn.rbm{u}.vis_units = opts.vis_units;
        else
            dbn.rbm{u}.vis_units = opts.hid_units;
        end;
        dbn.rbm{u}.hid_units = opts.hid_units;

        % weights
        dbn.rbm{u}.W  = normrnd(0, rand_weight_sigma, dbn.sizes(u + 1), dbn.sizes(u));
        dbn.rbm{u}.vW = zeros(dbn.sizes(u + 1), dbn.sizes(u));

        % visible biases
        dbn.rbm{u}.b  = zeros(dbn.sizes(u), 1);
        dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);

        % hidden biases
        dbn.rbm{u}.c  = zeros(dbn.sizes(u + 1), 1);
        dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1);
    end

end
