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
    if (~isfield(opts,'momentum') || isempty(opts.momentum))
        opts.momentum = 0;
    end;
    if (~isfield(opts,'momentum_first_rbm') || isempty(opts.momentum_first_rbm))
        opts.momentum_first_rbm = opts.momentum;
    end;
    if (~isfield(opts,'alpha') || isempty(opts.alpha))
        opts.alpha = 0.1;
    end;
    if (~isfield(opts,'alpha_first_rbm') || isempty(opts.alpha_first_rbm))
        opts.alpha_first_rbm = opts.alpha;
    end;
    if (~isfield(opts,'cdn') || isempty(opts.momentum))
        opts.cdn = 1;
    end;

    for u = 1 : numel(dbn.sizes) - 1
        dbn.rbm{u}.cdn      = opts.cdn;
        
        % make vis_units only actually visible units (1st layer),
        % set parameterst for first RBM
        if (u == 1)
            dbn.rbm{u}.vis_units = opts.vis_units;
            dbn.rbm{u}.momentum = opts.momentum_first_rbm;
            dbn.rbm{u}.alpha    = opts.alpha_first_rbm;
        else
            dbn.rbm{u}.vis_units = opts.hid_units;
            dbn.rbm{u}.alpha    = opts.alpha;
            dbn.rbm{u}.momentum = opts.momentum;
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
