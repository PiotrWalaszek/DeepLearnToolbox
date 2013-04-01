function res = logsumexpoverrows(a)
error('not working - use log_sum_over_cols.m')
maxs_small = max(a,[],1);
maxs_big = repmat(maxs_small,size(a,1), 1);
res = log( sum( exp( a-maxs_big), 1)) + maxs_small;
end
