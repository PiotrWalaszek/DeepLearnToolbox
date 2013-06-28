function [x, x_sample] = rbmup(rbm, x)
    x = repmat(rbm.c', size(x, 1), 1) + x * rbm.W'; % raw input
    switch rbm.hid_units
        case 'sigm'
            x = sigm(x);
            x_sample = sample(x);
        case 'linear'
            % no change, just raw input
            x_sample = x;
        case 'ReLUnoisy'
            x = ReLU(x + normrnd(0,1,size(x)));
            x_sample = x;
        otherwise
            error('Invalid hidden unit type (rbm.hid_unit)');
    end;
end
