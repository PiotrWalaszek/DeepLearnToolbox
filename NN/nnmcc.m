function [mcc, bad] = nnmcc(nn, x, y, threshold)

assert(size(y,2) == 1,'Behavior of matthew correlation tested only for single output. For > 1 output units use nnmatthew')

if ~exist('threshold','var')
   threshold = 0.5;
end

predicted_score = nnpredict(nn, x);
predicted_labels = predicted_score > threshold;

einfo = error_info( predicted_labels', y');
mcc = einfo.mcc;
bad = einfo.bad;
end