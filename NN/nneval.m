function [loss] = nneval(nn, loss, train_x, train_y, val_x, val_y)
%NNEVAL evaluates performance of neural network
% Returns a updated loss struct
assert(nargin == 4 || nargin == 6, 'Wrong number of arguments');

% training performance
nn           = nnff(nn, train_x, train_y);
loss.train.e = [loss.train.e; nn.L];

% validation performance
if nargin == 6
    nn           = nnff(nn, val_x, val_y);
    loss.val.e   = [loss.val.e; nn.L];
end

%If error function is supplied apply it
if isfield(nn, 'errfun')
    [er_train, ~]               = nn.errfun(nn, train_x, train_y);
    loss.train.e_errfun         = [loss.train.e_errfun; er_train];
    
    if nargin == 6
        [er_val, ~]             = nn.errfun(nn, val_x, val_y);
        loss.val.e_errfun      = [loss.val.e_errfun; er_val];
    end
end

end
