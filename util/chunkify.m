function chunks = chunkify(chunksize,data)
%%CHUNKIFY extract minibatch index
% return row indexes for chunks of a given size
[m,n] = size(data);
numchunks =ceil( m / chunksize);
batchstart = 1;
batchend   = chunksize;
chunks = zeros(2,numchunks);
for i = 1:numchunks
    if batchend <= m
       chunks(1,i)  = batchstart;
       chunks(2,i)  = batchend;
    else
       chunks(1,i)  = batchstart;
       chunks(2,i)  = m;
    end
    batchstart = batchend +1;
    batchend   = batchstart + chunksize -1;
end
end